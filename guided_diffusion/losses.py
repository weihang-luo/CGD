"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import os
from matplotlib import pyplot as plt
import numpy as np

import torch as th
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn
import time


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

#############################################################################

# 均方差
def mask_mse_loss(_x0, _pred_x0, _mask=None):
    if _mask is None:
        _mask = th.ones_like(_x0)

    return th.sum((_x0  * _mask - _pred_x0 * _mask) ** 2) 
 
# 计算原始损失
def original_loss(
        _x0, 
        _pred_x0, 
        _mask=None, 
        ):
    return mask_mse_loss(_x0, _pred_x0, _mask)

#############################################################################
# 拉普拉斯新损失函数
#############################################################################

# 蒙版可视化
def save_combined_images(_x0, _pred_crop0, mask, save_path, step):
    """
    保存目标图像（彩色）、原始生成图像（彩色）和蒙版（灰度）为组合图像
    参数：
        pred_crop0: 目标图像，形状为 (N, C, H, W)
        _x0: 原始生成图像，形状为 (N, C, H, W)
        mask: 显著性蒙版，形状为 (N, C, H, W)
        save_path: 保存路径
        step: 当前时间步
    """
    os.makedirs(save_path, exist_ok=True)

    # 获取当前时间戳
    timestamp = time.strftime("%M%S", time.localtime())

    # 处理批次内的每张图像
    batch_size = _pred_crop0.shape[0]
    combined_images = []

    for pred, x0, m in zip(_pred_crop0, _x0, mask):
        # 处理目标图像（彩色）[-1, 1] 转换到 [0, 1]
        pred_rgb = pred.permute(1, 2, 0).detach().cpu().numpy()  # 转换为 (H, W, C)
        pred_rgb = (pred_rgb + 1) / 2  # 将 [-1, 1] 映射到 [0, 1] 范围
        pred_rgb = np.clip(pred_rgb, 0, 1)

        # 处理原始生成图像（彩色）[-1, 1] 转换到 [0, 1]
        x0_rgb = x0.permute(1, 2, 0).detach().cpu().numpy()  # 转换为 (H, W, C)
        x0_rgb = (x0_rgb + 1) / 2  # 将 [-1, 1] 映射到 [0, 1] 范围
        x0_rgb = np.clip(x0_rgb, 0, 1)

        # 处理蒙版灰度图 [0, 1]
        mask_gray = m.mean(dim=0).detach().cpu().numpy()  # 平均 RGB 通道，转为灰度图
        mask_rgb = np.expand_dims(mask_gray, axis=-1).repeat(3, axis=-1)  # 灰度图扩展为伪彩色
        mask_rgb = np.clip(mask_rgb, 0, 1)  # 确保范围 [0, 1]

        # 拼接目标图像、原始生成图像和蒙版
        combined = np.vstack([
            (pred_rgb * 255).astype(np.uint8),  # 彩色目标图
            (x0_rgb * 255).astype(np.uint8),   # 彩色原始生成图像
            (mask_rgb * 255).astype(np.uint8)  # 灰度蒙版
        ])
        combined_images.append(combined)

    # 将批次内的图像拼接到一张图：每行 N 张
    row_images = [
        np.hstack(combined_images[i:i + batch_size])  # 每 N 张拼接为一行
        for i in range(0, batch_size, batch_size)
    ]
    final_image = np.vstack(row_images)  # 将所有行拼接为最终图像

    # 保存最终拼接图像
    filename = os.path.join(
        save_path, f"mask_step_{step}_{timestamp}.png"
    )
    plt.imsave(filename, final_image)


def laplacian_pyramid(img, levels=4):
    """
    生成拉普拉斯金字塔
    参数：
        img: 输入图像，形状为 (N, C, H, W)
        levels: 金字塔层数
    返回：
        包含各层拉普拉斯图像的列表
    """
    pyramid = []
    current_img = img
    for _ in range(levels):
        # 下采样
        downsampled = F.avg_pool2d(current_img, kernel_size=2, stride=2)
        # 上采样回原尺寸
        upsampled = F.interpolate(downsampled, size=current_img.shape[2:], mode='bilinear', align_corners=False)
        # 计算拉普拉斯层
        laplacian = current_img - upsampled
        pyramid.append(laplacian)
        # 更新当前图像
        current_img = downsampled
    # 最后一层的高斯金字塔（最低频）也添加到金字塔中
    pyramid.append(current_img)
    return pyramid

def laplacian_pyramid(img, levels=4):
    """
    生成拉普拉斯金字塔
    参数：
        img: 输入图像，形状为 (N, C, H, W)
        levels: 金字塔层数
    返回：
        包含各层拉普拉斯图像的列表
    """
    pyramid = []
    current_img = img
    for _ in range(levels):
        # 下采样
        downsampled = F.avg_pool2d(current_img, kernel_size=2, stride=2)
        # 上采样回原尺寸
        upsampled = F.interpolate(downsampled, size=current_img.shape[2:], mode='bilinear', align_corners=False)
        # 计算拉普拉斯层
        laplacian = current_img - upsampled
        pyramid.append(laplacian)
        # 更新当前图像
        current_img = downsampled
    # 最后一层的高斯金字塔（最低频）也添加到金字塔中
    pyramid.append(current_img)
    return pyramid

def dynamic_laplacian_loss(
    _x0, 
    _pred_x0, 
    al_cumd_t, 
    levels=4, 
    mask=None, 

):
    """
    动态拉普拉斯金字塔损失，基于最大值融合的累积蒙版
    
    参数：
        _x0 (torch.Tensor): 生成的图像，取值范围为 [-1, 1]
        _pred_x0 (torch.Tensor): 当前时间步的目标图像，取值范围为 [-1, 1]
        al_cumd_t (float): 当前累积时间步的参数
        levels (int, optional): 拉普拉斯金字塔的层数。默认为4。
        use_mask (bool, optional): 是否使用显著性蒙版。默认为False。
        accumulated_mask (torch.Tensor, optional): 累积蒙版。默认为None。
        save_path (str, optional): 保存路径，用于可视化。默认为None。
        step (int, optional): 当前步骤，用于决定是否保存可视化图像。默认为None。
        update_weight (bool, optional): 是否更新累积蒙版。默认为False。
    
    返回：
        total_loss (torch.Tensor): 总损失值，用于优化
        accumulated_mask (torch.Tensor): 更新后的累积蒙版
    """
    beta_t = 1.0 - al_cumd_t

    mask = th.ones_like(_x0) if mask is None else mask
    
    # Apply mask to inputs
    x0_masked = _x0 * mask
    pred_x0_masked = _pred_x0 * mask

    # Generate Laplacian pyramids for masked images
    pyramid_x0 = laplacian_pyramid(x0_masked, levels)
    pyramid_pred_x0 = laplacian_pyramid(pred_x0_masked, levels)

    num_layers = len(pyramid_x0)
    low_freq_weight = max(beta_t, 0.1)
    high_freq_weight = (1.0 - low_freq_weight) / (num_layers - 1) if num_layers > 1 else 1.0

    # Precompute layer weights to avoid repetitive conditional checks in the loop
    layer_weights = [high_freq_weight] * (num_layers - 1) + [low_freq_weight]

    # Compute total loss as weighted sum of MSE losses across pyramid levels
    total_loss = sum(
        weight * F.mse_loss(lap_x0, lap_pred_x0, reduction='sum')
        for lap_x0, lap_pred_x0, weight in zip(pyramid_x0, pyramid_pred_x0, layer_weights)
    ) / levels

    return total_loss


#############################################################################



