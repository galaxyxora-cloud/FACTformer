import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse


class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr

def format_params(num):
        # 完整格式：带千位分隔符
        full_format = f"{num:,}"
        # 简化格式：带K/M/G单位
        if num >= 1e9:  # 大于1e9用G单位
            return f"{full_format} ({num/1e9:.2f}G)"
        elif num >= 1e6:  # 大于1e6用M单位
            return f"{full_format} ({num/1e6:.2f}M)"
        elif num >= 1e3:  # 大于1e3用K单位
            return f"{full_format} ({num/1e3:.2f}K)"
        else:
            return full_format

def tensor_metric(img, imclean, model, data_range=1):  # 计算图像PSNR输入为Tensor

    img_cpu = img.data.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
    imgclean = imclean.data.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
    SUM = 0
    for i in range(img_cpu.shape[0]):

        if model == 'PSNR':
            SUM += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range)
        elif model == 'MSE':
            SUM += compare_mse(imgclean[i, :, :, :], img_cpu[i, :, :, :])
        elif model == 'SSIM':
            SUM += compare_ssim(imgclean[i, :, :, :], img_cpu[i, :, :, :],channel_axis=2, data_range=data_range)
        else:
            print('Model False!')
    return SUM / img_cpu.shape[0]