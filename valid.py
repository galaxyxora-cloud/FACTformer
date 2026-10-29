import torch
import torch.nn.functional as F_nn
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder, tensor_metric
import os
import numpy as np
import gc
import shutil


def _valid(model, args, ep, debug_mode=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ots = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()
    metrics_list = []


    black_image_count = 0
    low_quality_count = 0

    temp_dir = os.path.join(args.result_dir, f'temp_{ep}')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with torch.no_grad():
        print('Start Evaluation')
        if debug_mode:
            print('调试模式已启用，将显示详细的数值范围信息')

        for idx, data in enumerate(ots):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                input_img, label_img, name = data
                img_name = name[0] if isinstance(name, tuple) else name

                input_img = input_img.to(device, non_blocking=True)
                label_img = label_img.to(device, non_blocking=True)

                orig_h, orig_w = input_img.shape[-2:]
                pad_h = (4 - orig_h % 4) % 4
                pad_w = (4 - orig_w % 4) % 4

                if pad_h != 0 or pad_w != 0:
                    input_img_padded = F_nn.pad(input_img, (0, pad_w, 0, pad_h), mode='reflect')
                else:
                    input_img_padded = input_img

                pred = model(input_img_padded)

                if pad_h != 0 or pad_w != 0:
                    pred = pred[..., :orig_h, :orig_w]


                pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
                pred_clip = torch.clamp(pred, 0.0, 1.0)

                label_img = torch.nan_to_num(label_img, nan=0.0, posinf=1.0, neginf=0.0)
                label_img = torch.clamp(label_img, 0.0, 1.0)


                if debug_mode:
                    pmin, pmax = float(pred_clip.min()), float(pred_clip.max())
                    lmin, lmax = float(label_img.min()), float(label_img.max())
                    print(f'[Debug] pred range: [{pmin:.6f}, {pmax:.6f}] finite: {bool(torch.isfinite(pred_clip).all())}')
                    print(f'[Debug] label range: [{lmin:.6f}, {lmax:.6f}] finite: {bool(torch.isfinite(label_img).all())}')


                pred_mean = pred_clip.mean().item()
                if pred_mean < 0.01:
                    black_image_count += 1


                save_path = os.path.join(temp_dir, img_name)

                img_to_save = torch.nan_to_num(pred_clip, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                img_to_save = img_to_save.detach().cpu()


                img_to_save = img_to_save.squeeze(0)
                if img_to_save.dim() != 3 or img_to_save.shape[0] not in (1, 3):
                    if img_to_save.dim() == 2:
                        img_to_save = img_to_save.unsqueeze(0)
                    if img_to_save.shape[0] == 1:
                        img_to_save = img_to_save.repeat(3, 1, 1)

                pred_pil = F.to_pil_image(img_to_save, mode='RGB')
                pred_pil.save(save_path)

                psnr = tensor_metric(pred_clip, label_img, 'PSNR', data_range=1)
                ssim_val = tensor_metric(pred_clip, label_img, 'SSIM', data_range=1)

                psnr_adder(psnr)
                ssim_adder(ssim_val)
                metrics_list.append((img_name, psnr, ssim_val))

                # 简单质量统计
                if psnr < 5.0:
                    low_quality_count += 1

                print(f'\rProcessing {idx + 1}/{len(ots)}: {img_name} - PSNR={psnr:.2f}, SSIM={ssim_val:.4f}', end='')

                del pred, pred_clip, img_to_save
                if 'input_img_padded' in locals() and input_img_padded is not input_img:
                    del input_img_padded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()

            except Exception as e:
                print(f'\n处理 {img_name} 时发生错误: {e}')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                continue


    avg_psnr = psnr_adder.average()
    avg_ssim = ssim_adder.average()

    final_dir = os.path.join(args.result_dir, f'epoch{ep}-psnr{avg_psnr:.2f}-ssim{avg_ssim:.4f}')
    if os.path.exists(final_dir):
        try:
            shutil.rmtree(final_dir)
        except Exception as e:
            print(f'[Warn] 删除已存在结果目录失败：{e}，将追加写入该目录名的后缀。')
            final_dir = final_dir + '_new'
    os.rename(temp_dir, final_dir)

    metrics_file = os.path.join(final_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write('Image Name, PSNR (dB), SSIM\n')
        for img_name, psnr, ssim_val in metrics_list:
            f.write(f'{img_name}, {psnr:.4f}, {ssim_val:.6f}\n')
        f.write(f'\nAverage, {avg_psnr:.4f}, {avg_ssim:.6f}\n')
        f.write(f'\nBlack-like images (<0.01 mean): {black_image_count}\n')
        f.write(f'Low-quality images (PSNR<5dB): {low_quality_count}\n')

    print(f'\nEpoch {ep} evaluation completed:')
    print(f'Average PSNR: {avg_psnr:.2f} dB')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Results saved in: {final_dir}')
    print(f'Black-like images: {black_image_count}, Low-quality images: {low_quality_count}')

    model.train()
    return avg_psnr, avg_ssim