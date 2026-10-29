import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder, tensor_metric
from data import test_dataloader
import time
import torch.nn.functional as F_nn
import math


def _eval(model, args):
    """
   (mode='test')
    """
    print(f"Loading model from: {args.test_model}")
    if os.path.exists(args.test_model):
        checkpoint = torch.load(args.test_model)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: Model file {args.test_model} not found!")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)

    psnr_adder = Adder()
    ssim_adder = Adder()
    time_adder = Adder()

    metrics_list = []

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    print(f'Start Testing on {len(dataloader)} images...')
    print(f'Results will be saved to: {args.result_dir}')

    factor = 8

    with torch.no_grad():
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            if isinstance(name, (list, tuple)):
                img_name = name[0]
            else:
                img_name = str(name)

            input_img = input_img.to(device)
            label_img = label_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            start_time = time.time()


            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0

            if padh != 0 or padw != 0:
                input_padded = F_nn.pad(input_img, (0, padw, 0, padh), 'reflect')
            else:
                input_padded = input_img


            pred = model(input_padded)
            pred = pred[:, :, :h, :w]

            elapsed = time.time() - start_time
            time_adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)
            label_clip = torch.clamp(label_img, 0, 1)

            psnr_val = tensor_metric(pred_clip, label_clip, 'PSNR', data_range=1)
            ssim_val = tensor_metric(pred_clip, label_clip, 'SSIM', data_range=1)

            psnr_adder(psnr_val)
            ssim_adder(ssim_val)

            metrics_list.append((img_name, psnr_val, ssim_val, elapsed))

            if args.save_image:
                save_path = os.path.join(args.result_dir, img_name)
                pred_pil = pred_clip.squeeze(0).cpu()
                F.to_pil_image(pred_pil).save(save_path)

            print(f'{iter_idx + 1}/{len(dataloader)}: {img_name} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}')

    avg_psnr = psnr_adder.average()
    avg_ssim = ssim_adder.average()
    avg_time = time_adder.average()

    metrics_file = os.path.join(args.result_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write('Image Name, PSNR (dB), SSIM, Time (s)\n')
        for i_name, i_psnr, i_ssim, i_time in metrics_list:
            f.write(f'{i_name}, {i_psnr:.4f}, {i_ssim:.6f}, {i_time:.4f}\n')
        f.write(f'\nAverage, {avg_psnr:.4f}, {avg_ssim:.6f}, {avg_time:.4f}\n')

    print('\n================ Test Summary ================')
    print(f'Total Images: {len(dataloader)}')
    print(f'Average PSNR: {avg_psnr:.2f} dB')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Average Time: {avg_time:.4f} s')
    print(f'Metrics Saved: {metrics_file}')
    print('==============================================')