#
# Novel view synthesis metrics on the held-out test views (all views that are not inputs).
#   PSNR    full image
#   PSNR-F  foreground only (GT object mask)
#   SSIM / LPIPS(vgg) on the masked images
#
#   python metrics.py -m <out>/scan24/gs
#

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from utils.loss_utils import ssim
from utils.image_utils import psnr, psnr_mask
from lpipsPyTorch.modules.lpips import LPIPS


def load(path):
    return tf.to_tensor(Image.open(path)).unsqueeze(0)[:, :3].cuda()


def evaluate(model_path):
    test_dir = Path(model_path) / "test"
    lpips_vgg = LPIPS('vgg', '0.1').cuda()
    results = {}
    for method in sorted(os.listdir(test_dir)):
        method_dir = test_dir / method
        names = sorted(os.listdir(method_dir / "renders"))
        per_view = {k: [] for k in ["PSNR", "PSNR-F", "SSIM", "LPIPS"]}
        for name in tqdm(names, desc="Metric evaluation ({})".format(method)):
            render = load(method_dir / "renders" / name)
            gt = load(method_dir / "gt" / name)
            mask = load(method_dir / "masks_gt" / name)
            per_view["PSNR"].append(psnr(render, gt).item())
            per_view["PSNR-F"].append(psnr_mask(render, gt, mask).item())
            per_view["SSIM"].append(ssim(render * mask, gt * mask).item())
            per_view["LPIPS"].append(lpips_vgg(render * mask, gt * mask).item())
        results[method] = {k: float(torch.tensor(v).mean()) for k, v in per_view.items()}
        results[method]["num_views"] = len(names)
        print(method, json.dumps(results[method], indent=2))
    with open(os.path.join(model_path, "results.json"), 'w') as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser(description="Novel view metrics")
    parser.add_argument('--model_path', '-m', required=True, type=str)
    args = parser.parse_args()
    torch.cuda.set_device(torch.device("cuda:0"))
    with torch.no_grad():
        evaluate(args.model_path)
