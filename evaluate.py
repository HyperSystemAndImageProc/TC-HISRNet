import math
import numpy as np
from scipy.signal import convolve2d

import pdb


def PSNR(pred, gt):
    valid = gt - pred
    rmse = math.sqrt(np.mean(valid ** 2))

    if rmse == 0:
        return 100
    psnr = 20 * math.log10(1.0 / rmse)
    return psnr


def SSIM(pred, gt):
    ssim = 0
    for i in range(gt.shape[0]):
        ssim = ssim + compute_ssim(pred[i, :, :], gt[i, :, :])
    return ssim / gt.shape[0]


def SAM(pred, gt):
    eps = 2.2204e-16
    pred[np.where(pred == 0)] = eps
    gt[np.where(gt == 0)] = eps

    nom = sum(pred * gt)
    denom1 = sum(pred * pred) ** 0.5
    denom2 = sum(gt * gt) ** 0.5
    sam = np.real(np.arccos(nom.astype(np.float32) / (denom1 * denom2 + eps)))
    sam[np.isnan(sam)] = 0
    sam_sum = np.mean(sam) * 180 / np.pi
    return sam_sum


'''原文链接：https: // blog.csdn.net / RSstudent / article / details / 115098695'''


def CC(pred, gt):
    pred=pred.reshape(pred.size,order='C')#order:CFA
    gt=gt.reshape(gt.size,order='C')
    cc=np.corrcoef(pred,gt)[0,1]
    return cc


# def ERGAS(pred, gt, lr):
#
#     h = 30  # 高分辨率影像分辨率
#     l = 120  # 低分辨率影像分辨率
#         # 此处也可通过列数计算，此处只是完全按照定义来看
#
#     channels = pred.shape[2]
#
#     inner_sum = 0
#     for channel in range(channels):
#         band_img1 = pred[:, :, channel]
#         band_img2 = gt[:, :, channel]
#         band_img3 = lr[:, :, channel]
#
#         rmse_value = rmse(band_img1, band_img2)
#         m = np.mean(band_img3)
#         inner_sum += np.power((rmse_value / m), 2)
#     mean_sum = inner_sum / channels
#     ergas = 100 * (h / l) * np.sqrt(mean_sum)
#
#     return ergas

def ERGAS(sr_img, gt_img, resize_factor=4):
    """Error relative global dimension de synthesis (ERGAS)
    reference: https://github.com/amteodoro/SA-PnP-GMM/blob/9e8dffab223d88642545d1760669e2326efe0973/Misc/ERGAS.m
    """
    sr_img = sr_img.astype(np.float64)
    gt_img = gt_img.astype(np.float64)
    err = sr_img - gt_img
    ergas = 0
    for i in range(err.shape[2]):
        ergas += np.mean(err[:, :, i]**2) / (np.mean(sr_img[:, :, i]))**2
    ergas = (100. / float(resize_factor)) * np.sqrt(1. / err.shape[2] * ergas)
    return ergas


# def sre(pred, gt):  # 信号与重构误差比
#     pred = gt.astype(np.float32)
#
#     sre_final = []
#     for i in range(gt.shape[2]):
#         numerator = np.square(np.mean(gt[:, :, i]))
#         denominator = (np.linalg.norm(gt[:, :, i] - pred[:, :, i])) / (gt.shape[0] * gt.shape[1])
#         sre_final.append(numerator / denominator)
#     return 10 * np.log10(np.mean(sre_final))

# def calculate_rmse(sr_img, gt_img):
#     """Calculate the relative RMSE"""
#     sr_img = sr_img.astype(np.float64)
#     gt_img = gt_img.astype(np.float64)
#     rmse = np.sqrt(np.mean((sr_img - gt_img)**2))
#     return rmse

def RMSE(pred, gt):
    if len(pred.shape) == 3:
        channels = pred.shape[2]
    else:
        channels = 1
        pred = np.reshape(pred, (pred.shape[0], pred.shape[1], 1))
        gt = np.reshape(gt, (gt.shape[0], gt.shape[1], 1))
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    def single_rmse(img1, img2):
        diff = img1 - img2
        mse = np.mean(np.square(diff))
        return np.sqrt(mse)

    rmse_sum = 0
    for band in range(channels):
        pred_band_img = pred[:, :, band]
        gt_band_img = gt[:, :, band]
        rmse_sum += single_rmse(pred_band_img, gt_band_img)

    rmse = round(rmse_sum, 2)*0.01
    return rmse



def matlab_style_gauss2D(shape=np.array([11, 11]), sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    siz = (shape - np.array([1, 1])) / 2
    std = sigma
    eps = 2.2204e-16
    x = np.arange(-siz[1], siz[1] + 1, 1)
    y = np.arange(-siz[0], siz[1] + 1, 1)
    m, n = np.meshgrid(x, y)

    h = np.exp(-(m * m + n * n).astype(np.float32) / (2. * sigma * sigma))
    h[h < eps * h.max()] = 0
    sumh = h.sum()

    if sumh != 0:
        h = h.astype(np.float32) / sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=np.array([win_size, win_size]), sigma=1.5)
    window = window.astype(np.float32) / np.sum(np.sum(window))

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)).astype(np.float32) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


