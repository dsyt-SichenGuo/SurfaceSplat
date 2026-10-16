"""SDF stages of SurfaceSplat (built on Voxurf).

coarse:  python run.py --config configs/dtu/coarse.py --scene scan24 --datadir <data>/dtu_scan24 \
             --basedir <out>/dtu --normal_dir <data>/dtu_scan24/normals
fine:    python run.py --config configs/dtu/fine.py   --scene scan24 --datadir <data>/dtu_scan24 \
             --basedir <out>/dtu --gs_dir <out>/dtu/scan24/gs/novel_views
mesh:    python run.py --config configs/dtu/fine.py   --scene scan24 --datadir ... --basedir ... --mesh_from_sdf

Outputs are written to <basedir>/<scene>/{coarse,fine}/.
"""
import os, sys, copy, json, time, random, argparse
from shutil import copyfile
from tqdm import trange
import math
import mmengine
import numpy as np
import trimesh
import logging
import torch
import torch.nn.functional as F
from datetime import datetime

from lib import utils, dtu_eval, mobile_brick_eval
from lib.load_data import load_data
from lib.utils import get_root_logger


def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--scene", type=str, required=True,
                        help='scene name, used as the experiment name (e.g. scan24, aston)')
    parser.add_argument("--datadir", type=str, required=True,
                        help='scene directory with image/, mask/ and the camera file')
    parser.add_argument("--basedir", type=str, required=True,
                        help='output root; results go to <basedir>/<scene>/<stage>')
    parser.add_argument("--num_img", type=int, default=None,
                        help='number of sparse input views (default: data.num_img in the config)')
    parser.add_argument("--normal_dir", type=str, default='',
                        help='directory of monocular normal maps (required when weight_normal > 0)')
    parser.add_argument("--gs_dir", type=str, default='',
                        help='3DGS novel-view directory (renders/, masks/, camera_pose_new.npy) for the fine stage')
    parser.add_argument("--eval_gt_dir", type=str, default='',
                        help='GT root for mesh evaluation (DTU: folder with ObsMask/ and Points/; '
                             'MobileBrick: defaults to --datadir)')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--mesh_from_sdf", action='store_true',
                        help='extract (and evaluate) the mesh of a trained stage, no training')
    parser.add_argument("--mesh_resolution", type=int, default=512)
    parser.add_argument("--extract_color", action='store_true')
    parser.add_argument("--load_density_only", type=int, default=1)
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    return parser


def get_normal_loss(normal_pred, normal_gt, cos_reduce='fg'):
    """L1 + cosine loss between unit normals; rays outside the object mask are zero vectors.

    cos_reduce='fg' averages the cosine term over the rays inside the mask, 'all' over all rays
    (the masked-out rays then only add a constant). The paper uses 'all' on DTU and 'fg' on MobileBrick.
    """
    normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
    cos = 1. - torch.sum(normal_pred * normal_gt, dim=-1)
    if cos_reduce == 'all':
        cos = cos.mean()
    else:
        valid = torch.abs(normal_pred - normal_gt).sum(dim=-1) > 0
        cos = (cos * valid).sum() / valid.sum().clamp(min=1)
    return l1, cos


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    mode = dict(getattr(cfg.data, 'mode', dict()))
    mode.pop('train_all', None)
    data_dict = load_data(
        cfg.data, reso_level=cfg.reso_level, num_img=cfg.data.num_img, white_bg=cfg.data.white_bkgd,
        normal_dir=args.normal_dir if cfg.surf_train.get('weight_normal', 0) > 0 else None,
        gs_dir=args.gs_dir or None, **mode)

    kept_keys = {
        'hwf', 'HW', 'Ks', 'near', 'far',
        'i_train', 'i_val', 'i_test', 'irregular_shape',
        'poses', 'render_poses', 'images', 'scale_mats_np', 'masks', 'normals'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['masks'] = torch.FloatTensor(data_dict['masks'], device='cpu')
    if data_dict['normals'] is not None:
        data_dict['normals'] = torch.FloatTensor(data_dict['normals'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict

def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):
    logger.info("= "*10 + "Begin training state [ {} ]".format(stage) + " ="*10)
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, masks, normals = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'masks', 'normals'
        ]
    ]
    # monocular normal supervision (SurfaceSplat, coarse stage)
    weight_normal = cfg_train.get('weight_normal', 0.)
    use_normal = weight_normal > 0
    if use_normal:
        assert normals is not None, 'weight_normal > 0 requires --normal_dir'
        assert cfg_train.ray_sampler == 'random', 'the normal loss is implemented for the random ray sampler'
    print("Train idx", i_train, "\nTest idx", i_test)

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif getattr(cfg_train, 'ft_path', ''):
        reload_ckpt_path = cfg_train.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model
    model_kwargs = copy.deepcopy(cfg_model)
    scale_ratio = getattr(cfg_train, 'scale_ratio', 2)
    num_voxels = model_kwargs.pop('num_voxels')
    num_voxels_bg = model_kwargs.pop('num_voxels_bg', num_voxels)
    if len(cfg_train.pg_scale) and not args.render_only:
        deduce = (scale_ratio**len(cfg_train.pg_scale))
        num_voxels = int(num_voxels / deduce)
        num_voxels_bg = int(num_voxels_bg / deduce)
        logger.info("\n" + "+ "*10 + "start with {} resolution deduction".format(deduce) + " +"*10 + "\n")
    else:
        deduce = 1

    model = Model.Voxurf(
        xyz_min=xyz_min, xyz_max=xyz_max,
        num_voxels=num_voxels,
        num_voxels_bg=num_voxels_bg,
        mask_cache_path=coarse_ckpt_path,
        exppath=os.path.join(cfg.basedir, cfg.expname),
        **model_kwargs)
    if cfg_model.maskout_near_cam_vox:
        model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    model = model.to(device)

    # init optimizer
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

    load_density_from = getattr(cfg_train, 'load_density_from', '')
    load_sdf_from = getattr(cfg_train, 'load_sdf_from', '')
    if load_density_from and stage == 'surf':
        density_ckpt_path = os.path.join(cfg.basedir, load_density_from)
        if args.load_density_only:
            model = utils.load_grid_data(model, density_ckpt_path, deduce)
        else:
            reload_ckpt_path = density_ckpt_path

    if reload_ckpt_path is None:
        logger.info(f'scene_rep_reconstruction ({stage}): train from scratch')
        start = 0
    else:
        logger.info(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer, strict=False)
        logger.info("Restart from iteration {}, model sdf size: {}".format(start, model.sdf.grid.shape))
        if reload_ckpt_path.split('/')[-1].split('_')[0] != stage:
            start = 0

    if cfg_train.get('load_param', False):
        model, _, _ = utils.load_checkpoint(
            model, None, cfg_train.load_sdf_from, True, strict=False)

    # init sdf
    if load_sdf_from:
        if hasattr(model, 'init_sdf_from_sdf'):
            sdf_reduce = cfg_train.get('sdf_reduce', 1.0)
            if cfg_train.load_sdf_from == 'auto':
                # the fine stage starts from the (normal-supervised) coarse stage of the same scene
                cfg_train.load_sdf_from = os.path.join(cfg.basedir, cfg.expname0, 'coarse', 'surf_last.tar')
            logger.info("\n" + "+ "*10 + "load sdf from: " + cfg_train.load_sdf_from + "+"*10 + "\n")
            sdf0 = utils.load_grid_data(model, cfg_train.load_sdf_from, name='sdf', return_raw=True)
            model.init_sdf_from_sdf(sdf0, smooth=False, reduce=sdf_reduce)
            if cfg_train.get('load_bg_all', False):
                bg_density0 = utils.load_grid_data(model, cfg_train.load_sdf_from, name='bg_density', return_raw=True)
                model.init_bg_density_from_bg_density(bg_density0)
                utils.load_weight_by_name(model, cfg_train.load_sdf_from, name='bg')
            elif cfg_train.get('load_bg_density', False):
                bg_density0 = utils.load_grid_data(model, cfg_train.load_sdf_from, name='bg_density', return_raw=True)
                model.init_bg_density_from_bg_density(bg_density0)
        else:
            model = utils.load_grid_data(model, cfg_train.load_sdf_from, name='sdf')
            smooth = getattr(model, 'init_sdf_smooth', False)
            if smooth:
                model.sdf = model.smooth_conv(model.sdf)
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    elif args.sdf_mode != "density" and load_density_from:
        smooth = getattr(model, 'init_density_smooth', True)
        model.init_sdf_from_density(smooth=smooth, reduce=1)
        # have to recreate the optimizer
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }
    if use_normal:
        render_kwargs['render_grad'] = True  # needed for the rendered normals

    # init batch rays sampler
    def gather_training_rays():
        data_device = 'cpu' if cfg.data.load2gpu_on_the_fly else device
        rgb_tr_ori = images[i_train].to(data_device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = Model.get_training_rays_in_maskcache_sampling(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train],
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                model=model, render_kwargs=render_kwargs,
            )
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = Model.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = Model.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = Model.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        if cfg_train.ray_sampler == 'patch':
            # patch sampler contains lots of empty spaces, remove them.
            index_generator = Model.batch_indices_generator(len(rgb_tr), 1)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler
    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()
    if use_normal:
        data_device = 'cpu' if cfg.data.load2gpu_on_the_fly else device
        normal_tr = normals[i_train].to(data_device)
        mask_tr = masks[i_train].to(data_device)
        poses_tr = poses[i_train].to(data_device)
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            with torch.no_grad():
                model.density[cnt <= 2] = -100
        per_voxel_init()

    # GOGO
    psnr_lst = []
    weight_lst = []
    mask_lst = []
    bg_mask_lst = []
    weight_sum_lst = []
    weight_nonzero_lst = []
    s_val_lst = []
    time0 = time.time()
    logger.info("start: {} end: {}".format(1 + start, 1 + cfg_train.N_iters))

    for global_step in trange(1+start, 1+cfg_train.N_iters):

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            if hasattr(model, 'num_voxels_bg'):
                model.scale_volume_grid(model.num_voxels * scale_ratio, model.num_voxels_bg * scale_ratio)
            else:
                model.scale_volume_grid(model.num_voxels * scale_ratio)
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'patch':
            sel_b = batch_index_sampler()
            patch_size = cfg_train.N_patch
            sel_r_start = torch.randint(rgb_tr.shape[1] - patch_size, [1])
            sel_c_start = torch.randint(rgb_tr.shape[2] - patch_size, [1])
            sel_r, sel_c = torch.meshgrid(torch.arange(sel_r_start[0], sel_r_start[0] + patch_size),
                                          torch.arange(sel_c_start[0], sel_c_start[0] + patch_size))
            sel_r, sel_c = sel_r.reshape(-1), sel_c.reshape(-1)
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        elif cfg_train.ray_sampler == 'random':
            # indices live on the device of the ray buffers (CPU when load2gpu_on_the_fly)
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand], device=rgb_tr.device)
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand], device=rgb_tr.device)
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand], device=rgb_tr.device)
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
            if use_normal:
                target_normal = normal_tr[sel_b, sel_r, sel_c]
                target_mask = mask_tr[sel_b, sel_r, sel_c]
                c2w = poses_tr[sel_b]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
            if use_normal:
                target_normal = target_normal.to(device)
                target_mask = target_mask.to(device)
                c2w = c2w.to(device)

        # volume rendering
        render_result = model(rays_o, rays_d, viewdirs, global_step=global_step, **render_kwargs)

        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
        psnr = utils.mse2psnr(loss.detach()).item()

        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_cum'][...,-1].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss

        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target.unsqueeze(-2)).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum(-1).mean()
            loss += cfg_train.weight_rgbper * rgbper_loss

        if global_step>cfg_train.tv_from and global_step<cfg_train.tv_end and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                tv_terms = getattr(cfg_train, 'tv_terms', dict())
                sdf_tv, smooth_grad_tv = tv_terms['sdf_tv'], tv_terms['smooth_grad_tv']
                if smooth_grad_tv > 0:
                    loss += cfg_train.weight_tv_density * model.density_total_variation(sdf_tv=0, smooth_grad_tv=smooth_grad_tv)
                if getattr(cfg_train, 'ori_tv', False):
                    loss += cfg_train.weight_tv_density * model.density_total_variation(sdf_tv=sdf_tv, smooth_grad_tv=0)
                    weight_tv_k0 = getattr(cfg_train, 'weight_tv_k0')
                    if weight_tv_k0 > 0:
                        k0_tv_terms = getattr(cfg_train, 'k0_tv_terms', dict())
                        loss += cfg_train.weight_tv_k0 * model.k0_total_variation(**k0_tv_terms)
                    if getattr(tv_terms, 'bg_density_tv', 0):
                        loss += cfg_train.weight_tv_density * model.density_total_variation(sdf_tv=0, smooth_grad_tv=0, bg_density_tv=tv_terms['bg_density_tv'])
        if getattr(cfg_train, 'ori_tv', False) and cfg_train.get('weight_bg_tv_k0', 0) >0 and global_step>cfg_train.tv_from and global_step%cfg_train.tv_every==0 and global_step<cfg_train.tv_end:
            bg_k0_tv_terms = getattr(cfg_train, 'bg_k0_tv_terms', dict())
            loss += cfg_train.get('weight_bg_tv_k0', 0) * model.bg_k0_total_variation(**bg_k0_tv_terms)

        if getattr(cfg_train, 'weight_rgb0', 0.) > 0:
            loss += F.mse_loss(render_result['rgb_marched0'], target) * cfg_train.weight_rgb0

        # normal consistency loss: rendered normals (rotated into the camera frame) vs. monocular normals
        normal_l1 = normal_cos = None
        if use_normal and render_result.get('normal_marched') is not None:
            rot = c2w[..., :3, :3].permute(0, 2, 1).detach()
            normal_pred = (rot @ render_result['normal_marched'][..., None])[..., 0]
            # optionally restrict the normal loss to the object mask
            m = target_mask if cfg_train.get('normal_mask', True) else 1.
            normal_l1, normal_cos = get_normal_loss(normal_pred * m, target_normal * m,
                                                     cos_reduce=cfg_train.get('normal_cos_reduce', 'fg'))
            loss += weight_normal * normal_l1 + weight_normal * normal_cos

        loss.backward()
        # make sure that density has no grad
        if global_step>cfg_train.tv_from and global_step<cfg_train.tv_end and global_step%cfg_train.tv_every==0:
            if not getattr(cfg_train, 'ori_tv', False):
                if cfg_train.weight_tv_density>0:
                    tv_terms = getattr(cfg_train, 'tv_terms', dict())
                    sdf_tv = tv_terms['sdf_tv']
                    if sdf_tv > 0:
                        model.sdf_total_variation_add_grad(
                            cfg_train.weight_tv_density * sdf_tv / len(rays_o), global_step < cfg_train.tv_dense_before)
                    bg_density_tv = getattr(tv_terms, 'bg_density_tv', 0)
                    if bg_density_tv > 0:
                        model.bg_density_total_variation_add_grad(
                            cfg_train.weight_tv_density * bg_density_tv / len(rays_o), global_step < cfg_train.tv_dense_before)                                                                                   
                if cfg_train.weight_tv_k0 > 0:
                    model.k0_total_variation_add_grad(
                        cfg_train.weight_tv_k0 / len(rays_o), global_step < cfg_train.tv_dense_before)
                if getattr(cfg_train, 'weight_bg_tv_k0', 0) > 0:
                    model.bg_k0_total_variation_add_grad(
                        cfg_train.weight_bg_tv_k0 / len(rays_o), global_step < cfg_train.tv_dense_before)
        optimizer.step()
        wm = render_result['weights'].max(-1)[0]
        ws = render_result['weights'].sum(-1)
        if (wm>0).float().mean() > 0:
            psnr_lst.append(psnr)
            weight_lst.append(wm[wm>0].mean().detach().cpu().numpy())
            weight_sum_lst.append(ws[ws>0].mean().detach().cpu().numpy())
            weight_nonzero_lst.append((ws>0).float().mean().detach().cpu().numpy())
            mask_lst.append(render_result['mask'].float().mean().detach().cpu().numpy())
            if 'bg_mask' in render_result:
                bg_mask_lst.append(render_result['bg_mask'].float().mean().detach().cpu().numpy())
        s_val = render_result["s_val"] if "s_val" in render_result else 0
        s_val_lst.append(s_val)

        global_step_ = global_step - 1
        # update lr
        N_iters = cfg_train.N_iters
        if not getattr(cfg_train, 'cosine_lr', ''):
            decay_steps = cfg_train.lrate_decay * 1000
            decay_factor = 0.1 ** (1/decay_steps)
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = param_group['lr'] * decay_factor
        else:
            def cosine_lr_func(iter, warm_up_iters, warm_up_min_ratio, max_steps, const_warm_up=False, min_ratio=0):
                if iter < warm_up_iters:
                    if not const_warm_up:
                        lr = warm_up_min_ratio + (1 - warm_up_min_ratio) * (iter / warm_up_iters)
                    else:
                        lr = warm_up_min_ratio
                else:
                    lr = (1 + math.cos((iter - warm_up_iters) / (max_steps - warm_up_iters) * math.pi)) * 0.5 * (1 - min_ratio) + min_ratio
                return lr
            def extra_warm_up_func(iter, start_iter, warm_up_iters, warm_up_min_ratio):
                if iter >= start_iter:
                    extra_lr = warm_up_min_ratio + (1 - warm_up_min_ratio) * (iter - start_iter) / warm_up_iters
                    return min(extra_lr, 1.0)
                else:
                    return 1.0
            warm_up_iters = cfg_train.cosine_lr_cfg.get('warm_up_iters', 0) 
            warm_up_min_ratio = cfg_train.cosine_lr_cfg.get('warm_up_min_ratio', 1.0)
            const_warm_up = cfg_train.cosine_lr_cfg.get('const_warm_up', False)
            cos_min_ratio = cfg_train.cosine_lr_cfg.get('cos_min_ratio', False)
            if global_step == 0:
                pre_decay_factor = 1.0
            else:
                pre_decay_factor = cosine_lr_func(global_step_ - 1, warm_up_iters, warm_up_min_ratio, N_iters, const_warm_up, cos_min_ratio)
            pos_decay_factor = cosine_lr_func(global_step_, warm_up_iters, warm_up_min_ratio, N_iters, const_warm_up, cos_min_ratio)
            decay_factor = pos_decay_factor / pre_decay_factor
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = param_group['lr'] * decay_factor
        decay_step_module = getattr(cfg_train, 'decay_step_module', dict())
        if global_step_ in decay_step_module:
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                if param_group['name'] in decay_step_module[global_step_]:
                    decay_factor = decay_step_module[global_step_][param_group['name']]
                    param_group['lr'] = param_group['lr'] * decay_factor
                    logger.info('- '*10 + '[Decay lrate] for {} by {}'.format(param_group['name'], decay_factor) + ' -'*10)

        # update tv terms
        tv_updates = getattr(cfg_train, 'tv_updates', dict())
        if global_step_ in tv_updates:
            for tv_term, value in tv_updates[global_step_].items():
                setattr(cfg_train.tv_terms, tv_term, value)
            logger.info('- '*10 + '[Update tv]: ' + str(tv_updates[global_step_]) + ' -'*10)

        # update s_val func
        s_updates = getattr(cfg_model, 's_updates', dict())
        if global_step_ in s_updates:
            for s_term, value in s_updates[global_step_].items():
                setattr(model, s_term, value)
            logger.info('- '*10 + '[Update s]: ' + str(s_updates[global_step_]) + ' -'*10)

        # update smooth kernel
        smooth_updates = getattr(cfg_model, 'smooth_updates', dict())
        if global_step_ in smooth_updates:
            model.init_smooth_conv(**smooth_updates[global_step_])
            logger.info('- '*10 + '[Update smooth conv]: ' + str(smooth_updates[global_step_]) + ' -'*10)

        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            bg_mask_mean = 0. if len(bg_mask_lst) == 0 else np.mean(bg_mask_lst)
            normal_str = '' if normal_l1 is None else f'Normal L1/cos: {normal_l1.item():.4f}/{normal_cos.item():.4f} / '
            logger.info(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                        f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / ' + normal_str +
                        f'Wmax: {np.mean(weight_lst):5.2f} / Wsum: {np.mean(weight_sum_lst):5.2f} / W>0: {np.mean(weight_nonzero_lst):5.2f}'
                        f' / s_val: {np.mean(s_val_lst):5.2g} / mask %: {100*np.mean(mask_lst):1.2f} / bg_mask %: {100*bg_mask_mean:1.2f} '
                        f'Eps: {eps_time_str}')
            psnr_lst, weight_lst, weight_sum_lst, weight_nonzero_lst, mask_lst, bg_mask_lst, s_val_lst = [], [], [], [], [], [], []

        # save checkpoints
        if global_step == cfg_train.N_iters:
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'MaskCache_kwargs': model.get_MaskCache_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, last_ckpt_path)
            logger.info(f'scene_rep_reconstruction ({stage}): saved checkpoints at '+ last_ckpt_path)


def train(args, cfg, data_dict):

    # init
    logger.info('train: start')
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # without object masks the scene is bounded by a fixed box in the normalized space
    box_size_ = cfg.surf_train.get('box_size', 1.5)
    print(">>> box_size: ", box_size_)
    xyz_min_fine = torch.tensor([-box_size_, -box_size_, -box_size_]).cuda()
    xyz_max_fine = torch.tensor([box_size_, box_size_, box_size_]).cuda()

    eps_surf = time.time()
    scene_rep_reconstruction(
        args=args, cfg=cfg,
        cfg_model=cfg.surf_model_and_render, cfg_train=cfg.surf_train,
        xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
        data_dict=data_dict, stage='surf')
    eps_surf = time.time() - eps_surf
    eps_time_str = f'{eps_surf//3600:02.0f}:{eps_surf//60%60:02.0f}:{eps_surf%60:02.0f}'
    logger.info("+ "*10 + 'train: {} stage finished in '.format(cfg.exp_stage) + eps_time_str + " +"*10)


@torch.no_grad()
def extract_and_evaluate_mesh(model, scale_mats_np, resolution=512, extract_color=False):
    '''Extract the zero level set, save it in world space (<scene>_mesh.ply) and in the
    normalized space (<scene>_mesh_normalized.ply, used for point sampling), then evaluate.
    '''
    mesh_dir = os.path.join(cfg.basedir, cfg.expname, 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)
    bound_min = model.xyz_min.clone().detach().float()
    bound_max = model.xyz_max.clone().detach().float()
    vertices0, triangles = model.extract_geometry(bound_min, bound_max, resolution=resolution,
                                                  threshold=0.0, scale_mats_np=scale_mats_np, smooth=True)
    vertices = vertices0 * scale_mats_np[0, 0] + scale_mats_np[:3, 3][None]

    if extract_color:
        # use normal direction as the viewdir
        ray_pts = torch.from_numpy(vertices0).cuda().float().split(8192 * 32, 0)
        vertex_colors = [model.mesh_color_forward(pts) for pts in ray_pts]
        vertex_colors = (torch.cat(vertex_colors).cpu().detach().numpy() * 255.).astype(np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
    else:
        mesh = trimesh.Trimesh(vertices, triangles)
    mesh_path = os.path.join(mesh_dir, '{}_mesh.ply'.format(args.scene))
    mesh.export(mesh_path)
    trimesh.Trimesh(vertices0, triangles).export(os.path.join(mesh_dir, '{}_mesh_normalized.ply'.format(args.scene)))
    logger.info("mesh saved at " + mesh_path)

    dataset = cfg.data.dataset_type
    results = None
    if dataset == 'dtu' and args.eval_gt_dir:
        scan_id = int(''.join(c for c in args.scene if c.isdigit()))
        mean_d2s, mean_s2d, over_all = dtu_eval.eval(
            mesh_path, scene=scan_id, eval_dir=mesh_dir, dataset_dir=args.eval_gt_dir,
            suffix='_eval', use_o3d=True, runtime=False)
        results = dict(d2s=float(mean_d2s), s2d=float(mean_s2d), chamfer=float(over_all))
    elif dataset == 'mobilebrick':
        gt_dir = args.eval_gt_dir or cfg.data.datadir
        if os.path.exists(os.path.join(gt_dir, 'mesh', 'gt_mesh.ply')):
            results = mobile_brick_eval.eval(mesh_path, gt_dir)
    if results is not None:
        logger.info('mesh evaluation: ' + json.dumps(results))
        with open(os.path.join(mesh_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    else:
        logger.info('no GT given, skip mesh evaluation')


if __name__=='__main__':
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmengine.Config.fromfile(args.config)
    cfg.data.datadir = args.datadir
    cfg.basedir = args.basedir
    if args.num_img is not None:
        cfg.data.num_img = args.num_img
    args.sdf_mode = 'voxurf_womask_' + cfg.exp_stage
    args.render_only = args.mesh_from_sdf

    cfg.expname0 = args.scene
    cfg.expname = os.path.join(args.scene, cfg.exp_stage)
    log_dir = os.path.join(cfg.basedir, cfg.expname, 'log')
    os.makedirs(log_dir, exist_ok=True)
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logger = get_root_logger(logging.INFO, handlers=[
        logging.FileHandler(os.path.join(log_dir, '{}_train.log').format(time_str))])
    logger.info("+ "*10 + cfg.expname + " +"*10)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    if cfg.exp_stage == 'coarse':
        import lib.voxurf_womask_coarse as Model
    elif cfg.exp_stage == 'fine':
        import lib.voxurf_womask_fine as Model
    else:
        raise NameError(cfg.exp_stage)

    # keep a copy of the code / config that produced the results
    rec_dir = os.path.join(cfg.basedir, cfg.expname, 'recording')
    os.makedirs(rec_dir, exist_ok=True)
    root = os.path.dirname(os.path.abspath(__file__))
    if not args.mesh_from_sdf:
        copyfile(os.path.join(root, 'run.py'), os.path.join(rec_dir, 'run.py'))
        copyfile(args.config, os.path.join(rec_dir, os.path.basename(args.config)))
        copyfile(Model.__file__, os.path.join(rec_dir, os.path.basename(Model.__file__)))

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    if args.mesh_from_sdf:
        logger.info('Extracting mesh from sdf...')
        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'surf_last.tar')
        model = utils.load_model(Model.Voxurf, ckpt_path, cfg.surf_model_and_render).to(device)
        extract_and_evaluate_mesh(model, data_dict['scale_mats_np'], resolution=args.mesh_resolution,
                                  extract_color=args.extract_color)
        logger.info('done')
        sys.exit()

    train(args, cfg, data_dict)
    logger.info('Done')
