"""Monocular normal estimation with Metric3D v2 (ViT-S), used by the coarse SDF stage.

For every image in <datadir>/image, saves a (3, H, W) float32 camera-space normal map to
<out_dir>/<image name>.npy (default <out_dir> = <datadir>/normals).

    python tools/estimate_normals.py --datadir <data>/dtu_scan24

The model is fetched with torch.hub from https://github.com/YvanYin/Metric3D.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

INPUT_SIZE = (616, 1064)  # input resolution of the ViT models
MEAN = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
STD = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
PADDING = [123.675, 116.28, 103.53]


@torch.no_grad()
def predict_normal(model, rgb_origin):
    """rgb_origin: (H, W, 3) uint8 RGB image -> (3, H, W) normal map."""
    h, w = rgb_origin.shape[:2]
    scale = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    h, w = rgb.shape[:2]
    pad_h, pad_w = INPUT_SIZE[0] - h, INPUT_SIZE[1] - w
    pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                             cv2.BORDER_CONSTANT, value=PADDING)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - MEAN), STD)[None].cuda()
    _, _, output_dict = model.inference({'input': rgb})

    pred_normal = output_dict['prediction_normal'][:, :3, :, :].squeeze()
    pred_normal = pred_normal[:, pad_info[0]: pred_normal.shape[1] - pad_info[1],
                              pad_info[2]: pred_normal.shape[2] - pad_info[3]]
    pred_normal = torch.nn.functional.interpolate(pred_normal[None], rgb_origin.shape[:2], mode='bilinear').squeeze()
    return pred_normal.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datadir', required=True, help='scene directory containing image/')
    parser.add_argument('--out_dir', default='', help='default: <datadir>/normals')
    parser.add_argument('--model', default='metric3d_vit_small')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.datadir, 'normals')
    os.makedirs(out_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(args.datadir, 'image', '*.png')) +
                         glob.glob(os.path.join(args.datadir, 'image', '*.jpg')))
    assert len(image_paths) > 0, f'no images in {args.datadir}/image'

    model = torch.hub.load('yvanyin/metric3d', args.model, pretrain=True).cuda().eval()
    for path in tqdm(image_paths, desc='Metric3D normals'):
        rgb = cv2.imread(path)[:, :, ::-1].copy()
        normal = predict_normal(model, rgb)
        np.save(os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + '.npy'), normal)
    print(f'saved {len(image_paths)} normal maps to {out_dir}')


if __name__ == '__main__':
    main()
