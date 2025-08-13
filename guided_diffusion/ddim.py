from functools import partial
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch.utils.checkpoint import checkpoint as grad_ckpt
from tqdm import tqdm

from utils.logger import logging_info
from .gaussian_diffusion import _extract_into_tensor
from .new_scheduler import ddim_timesteps, ddim_repaint_timesteps
from .respace import SpacedDiffusion
from .losses import  original_loss, dynamic_laplacian_loss, save_combined_images


def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()

def get_g_scha(length = 250, start = 1, end = 10,  stop=0):
    # 更精确的步长计算方法，考虑浮点数后再四舍五入
    step = ((end - start) / (length - stop - 1)) 

    # 生成序列，通过累积和四舍五入来逼近均匀分布
    sequence = [round(start + step * i) for i in range(length - stop)]

    # 确保序列最后一个值为10
    if sequence[-1] != end:
        # 如果最后一个值不是目标值，适当调整以确保序列正确结束
        diff = end - sequence[-1]
        # 将差值均匀分配到前面的某些元素上，这里简化处理，直接调整最后一个值
        sequence[-1] = end
    
    if stop:
        sequence = [0] * stop + sequence

    return sequence

# 
def split_list_with_overlap(original_list, sublist_length):

    """
    将原始列表分割成多个子列表，确保每个子列表与前一个子列表有一个重叠元素。
    
    参数:
    original_list (list): 需要分割的原始列表。
    sublist_length (int): 每个子列表的长度（不包括重叠元素）。
    
    返回:
    list of lists: 分割后的子列表，每个子列表长度为 sublist_length + 1。
    """

    sublists = []
    # 添加第一个子列表
    sublists.append(original_list[:sublist_length])
    
    # 处理剩余的列表部分，从第二段开始
    for i in range(sublist_length, len(original_list), sublist_length):
        # 前一个子列表的最后一个元素
        previous_end = sublists[-1][-1]
        # 当前子列表，从i-1开始
        sublist = [previous_end] + original_list[i:i + sublist_length]
        sublists.append(sublist)
    
    return sublists


def create_custom_gaussian_mask(
    center: Tuple[float, float],
    size: int,
    radius: float,
    falloff_factor: float,
    channels: int = 3,
    batch_size: int = 4,
    device: torch.device = torch.device('cpu'),
    mode: str = 'mask1'
) -> torch.Tensor:
    """
    Create a custom Gaussian mask using PyTorch.

    Args:
        center (Tuple[float, float]): The (x, y) center of the mask.
        size (int): The size (width and height) of the mask.
        radius (float): The radius for mask creation.
        falloff_factor (float): Determines the falloff behavior.
        channels (int, optional): Number of channels. Defaults to 3.
        batch_size (int, optional): Batch size. Defaults to 4.
        device (torch.device, optional): Device to create the tensor on. Defaults to 'cpu'.
        mode (str, optional): Mode of mask creation ('mask1' or 'mask2'). Defaults to 'mask1'.

    Returns:
        torch.Tensor: The generated mask tensor of shape (batch_size, channels, size, size).
    """
    # 创建坐标网格
    x = torch.linspace(0, size - 1, steps=size, device=device)
    y = torch.linspace(0, size - 1, steps=size, device=device)
    y, x = torch.meshgrid(y, x)  # Shape: (size, size)

    # 计算距离中心的距离
    distance_from_center = torch.sqrt((x - center[0])**2 + (y - center[1])**2)
    center_mask = distance_from_center <= radius  # Shape: (size, size)
 
    if mode == 'mask1':
        # Mode 1: Hard center with Gaussian falloff
        if falloff_factor == 0.0:
            gaussian = torch.ones_like(distance_from_center, device=device)
        else:
            # 确保传递给 torch.log 的参数为 Tensor
            #####################################################################################################
            log_input = torch.log(torch.tensor(1 + falloff_factor, device=device))
            #####################################################################################################
            std = radius / torch.sqrt(2 * log_input)
            gaussian = torch.exp(-((distance_from_center - radius)**2) / (2 * std**2))
            mask = torch.where(center_mask, torch.ones_like(distance_from_center, device=device), gaussian)
    
    elif mode == 'mask2':
        # Mode 2: Smooth transition based on falloff_factor using Sigmoid
        # radius = int(size* 0.5 * falloff_factor)
        radius = max(int(size* 0.45), int(size* 0.5 * falloff_factor))
        center_mask = distance_from_center <= radius  # Shape: (size, size)
        threshold = 0.03  # 设置阈值
        k = 20  # Sigmoid 函数的陡峭度，可以根据需要调整
        
        # 计算权重 alpha
        alpha = torch.sigmoid(k * (torch.tensor(falloff_factor, device=device) - torch.tensor(threshold, device=device)))

        if falloff_factor == 0.0:
            gaussian = torch.ones_like(distance_from_center, device=device)
        else:
            #####################################################################################################
            ft = 2- falloff_factor
            # 确保传递给 torch.log 的参数为 Tensor
            log_input = torch.log(torch.tensor(ft, device=device))
            #####################################################################################################
            std = radius / torch.sqrt(2 * log_input)
            gaussian = torch.exp(-((distance_from_center - radius)**2) / (2 * std**2))
            mask = torch.where(center_mask, torch.ones_like(distance_from_center, device=device), gaussian)
            mask = 1 -  mask * alpha

    else:
        raise ValueError("Invalid mode. Choose 'mask1' or 'mask2'.")

    # 扩展掩码到通道和批量
    mask = mask.unsqueeze(0).repeat(channels, 1, 1)  # Shape: (channels, size, size)
    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # Shape: (batch_size, channels, size, size)

    return mask

def generate_sequence(lenth, subsequences, falloffs):
    """
    生成一个序列，长度为 lenth

    参数:
    lenth (int): 序列长度。
    subsequences (list): 
    falloffs (list): 
    
    返回:
    list: 生成的序列。
    """
    assert sum(subsequences) == lenth, " sum subsequences should be lenth."
    assert len(subsequences) == len(falloffs)

    sequence = []
    for subseq, falloff in zip(subsequences, falloffs):
        sequence.extend([falloff] * subseq)
    
    return sequence[::-1]

def create_steep_to_gentle_sequence(length, position, a=0.1):
    """
    根据给定的序列长度和位置序号，返回该位置上的值。
    序列的值从小到大变化，且呈现出前陡后缓的趋势。

    参数:
    length (int): 序列的长度。
    position (int): 序列中的位置序号（从0开始）。
    a (float): 衰减速率，较小的 a 值会使曲线更平缓。

    返回:
    float: 指定位置上的值。
    """
    t = np.linspace(0, length - 1, length)
    sequence = np.exp(-a * t)
    sequence_normalized = (sequence.max() - sequence) / (sequence.max() - sequence.min())
    return sequence_normalized[position] 
    
def gen_repair_seq(n):
    if n < 0:
        raise ValueError("The specified number must be non-negative.")
    
    # Create the increasing sequence from 0 to n
    increasing_sequence = list(range(n + 1))
    # Create the decreasing sequence from n-1 to 0
    decreasing_sequence = list(range(n - 1, -1, -1))
    
    # Concatenate both sequences
    full_sequence = increasing_sequence + decreasing_sequence
    return full_sequence

class DDIMSampler(SpacedDiffusion):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )
        self.ddim_sigma = conf.get("ddim.ddim_sigma")

    def _get_et(self, model_fn, x, t, model_kwargs):
        model_fn = self._wrap_model(model_fn)
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, _ = torch.split(model_output, C, dim=1)
        return model_output

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self,
        model_fn,
        x,
        t,
        prev_t,
        model_kwargs,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        with torch.no_grad():
            alpha_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, prev_t, x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )

            def process_xstart(_x):
                if denoised_fn is not None:
                    _x = denoised_fn(_x)
                if clip_denoised:
                    return _x.clamp(-1, 1)
                return _x

            e_t = self._get_et(model_fn, x, t, model_kwargs)
            pred_x0 = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=e_t))

            mean_pred = (
                pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * e_t
            )
            noise = noise_like(x.shape, x.device, repeat=False)

            nonzero_mask = (t != 0).float().view(-1, *
                                                ([1] * (len(x.shape) - 1)))
            x_prev = mean_pred + noise * sigmas * nonzero_mask

        return {
            "x_prev": x_prev,
            "pred_x0": pred_x0,
        }

    def q_sample_middle(self, x, cur_t, tar_t, no_noise=False):
        assert cur_t <= tar_t
        device = x.device
        while cur_t < tar_t:
            if no_noise:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x)
            _cur_t = torch.tensor(cur_t, device=device)
            beta = _extract_into_tensor(self.betas, _cur_t, x.shape)
            x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise
            cur_t += 1
        return x

    def q_sample(self, x_start, t, no_noise=False):
        if no_noise:
            noise = torch.zeros_like(x_start)
        else:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod,
                                t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def x_forward_sample(self, x0, forward_method="from_0", no_noise=False):
        x_forward = [self.q_sample(x0, torch.tensor(0, device=x0.device))]
        if forward_method == "from_middle":
            for _step in range(0, len(self.timestep_map) - 1):
                x_forward.append(
                    self.q_sample_middle(
                        x=x_forward[-1][0].unsqueeze(0),
                        cur_t=_step,
                        tar_t=_step + 1,
                        no_noise=no_noise,
                    )
                )
        elif forward_method == "from_0":
            for _step in range(1, len(self.timestep_map)):
                x_forward.append(
                    self.q_sample(
                        x_start=x0,#[0].unsqueeze(0),
                        t=torch.tensor(_step, device=x0.device),
                        no_noise=no_noise,
                    )
                )
        return x_forward

    def p_sample_loop(
        self,
        model_fn,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None,
        sample_dir="",
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(shape, device=device)

        assert conf["ddim.schedule_params"] is not None
        steps = ddim_timesteps(**conf["ddim.schedule_params"])  # 获取倒序steps列表
        time_pairs = list(zip(steps[:-1], steps[1:]))

        x0 = model_kwargs["gt"]
        x_forwards = self.x_forward_sample(x0, "from_0")  # 获取X0到每一步加噪图像
        mask = model_kwargs["gt_keep_mask"]

        x_t = img
        import os
        from utils import normalize_image, save_grid

        for cur_t, prev_t in tqdm(time_pairs):
            # replace surrounding
            x_t = x_forwards[cur_t] * mask + (1.0 - mask) * x_t
            cur_t = torch.tensor([cur_t] * shape[0], device=device)
            prev_t = torch.tensor([prev_t] * shape[0], device=device)

            output = self.p_sample(
                model_fn,
                x=x_t,
                t=cur_t,
                prev_t=prev_t,
                model_kwargs=model_kwargs,
                conf=conf,
                pred_xstart=None,
            )
            x_t = output["x_prev"]

            if conf["debug"]:
                from utils import normalize_image, save_grid

                os.makedirs(os.path.join(sample_dir, "middles"), exist_ok=True)
                save_grid(
                    normalize_image(x_t),
                    os.path.join(sample_dir, "middles",
                                f"mid-{prev_t[0].item()}.png"),
                )
                save_grid(
                    normalize_image(output["pred_x0"]),
                    os.path.join(sample_dir, "middles",
                                f"pred-{prev_t[0].item()}.png"),
                )

        x_t = x_t.clamp(-1.0, 1.0)
        return {
            "sample": x_t,
        }


class R_DDIMSampler(DDIMSampler):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )

    @staticmethod
    def resample(m, w, n):
        """
        m: max number of index
        w: un-normalized probability
        n: number of indices to be selected
        """
        if max([(math.isnan(i) or math.isinf(i)) for i in w]):
            w = np.ones_like(w)
        if w.sum() < 1e-6:
            w = np.ones_like(w)

        w = n * (w / w.sum())
        c = [int(i) for i in w]
        r = [i - int(i) for i in w]
        added_indices = []
        for i in range(m):
            for j in range(c[i]):
                added_indices.append(i)
        if len(added_indices) != n:
            R = n - sum(c)
            indices_r = torch.multinomial(torch.tensor(r), R)
            for i in indices_r:
                added_indices.append(i)
        logging_info(
            "Indices after Resampling: %s"
            % (" ".join(["%.d" % i for i in sorted(added_indices)]))
        )
        return added_indices

    @staticmethod
    def gaussian_pdf(x, mean, std=1):
        return (
            1
            / (math.sqrt(2 * torch.pi) * std)
            * torch.exp(-((x - mean) ** 2).sum() / (2 * std**2))
        )

    def resample_based_on_x_prev(
        self,
        x_t,
        x_prev,
        x_pred_prev,
        mask,
        keep_n_samples=None,
        temperature=100,
        p_cal_method="mse_inverse",
        pred_x0=None,
    ):
        if p_cal_method == "mse_inverse":  # same intuition but empirically better
            mse = torch.tensor(
                [((x_prev * mask - i * mask) ** 2).sum() for i in x_pred_prev]
            )
            mse /= mse.mean()
            p = torch.softmax(temperature / mse, dim=-1)
        elif p_cal_method == "gaussian":
            p = torch.tensor(
                [self.gaussian_pdf(x_prev * mask, i * mask)
                for i in x_pred_prev]
            )
        else:
            raise NotImplementedError
        resample_indices = self.resample(
            x_t.shape[0], p, x_t.shape[0] if keep_n_samples is None else keep_n_samples
        )
        x_t = torch.stack([x_t[i] for i in resample_indices], dim=0)
        x_pred_prev = torch.stack([x_pred_prev[i]
                                for i in resample_indices], dim=0)
        pred_x0 = (
            torch.stack([pred_x0[i] for i in resample_indices], dim=0)
            if pred_x0 is not None
            else None
        )
        logging_info(
            "Resampling with probability %s" % (
                " ".join(["%.3lf" % i for i in p]))
        )
        return x_t, x_pred_prev, pred_x0

    def p_sample_loop(
        self,
        model_fn,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None,
        sample_dir="",
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(shape, device=device)

        assert conf["ddim.schedule_params"] is not None
        steps = ddim_timesteps(**conf["ddim.schedule_params"])
        time_pairs = list(zip(steps[:-1], steps[1:]))
        
        x0 = model_kwargs["gt"]
        mask = model_kwargs["gt_keep_mask"]
        # x_forwards = self.x_forward_sample(x0, "from_middle")
        x_forwards = self.x_forward_sample(x0, "from_0")
        x_t = img

        for cur_t, prev_t in tqdm(time_pairs):
            x_t = x_forwards[cur_t] * mask + (1.0 - mask) * x_t
            x_prev = x_forwards[prev_t]
            output = self.p_sample(
                model_fn,
                x=x_t,
                t=torch.tensor([cur_t] * shape[0], device=device),
                prev_t=torch.tensor([prev_t] * shape[0], device=device),
                model_kwargs=model_kwargs,
                conf=conf,
                pred_xstart=None,
            )

            x_pred_prev, x_pred_x0 = output["x_prev"], output["pred_x0"]
            x_t, x_pred_prev, pred_x0 = self.resample_based_on_x_prev(
                x_t=x_t,
                x_prev=x_prev,
                x_pred_prev=x_pred_prev,
                mask=mask,
                pred_x0=x_pred_x0,
            )
            if conf["debug"]:
                from utils import normalize_image, save_grid

                os.makedirs(os.path.join(sample_dir, "middles"), exist_ok=True)
                save_grid(
                    normalize_image(x_t),
                    os.path.join(sample_dir, "middles", f"mid-{prev_t}.png"),
                )
                save_grid(
                    normalize_image(pred_x0),
                    os.path.join(sample_dir, "middles", f"pred-{prev_t}.png"),
                )

        x_t = self.resample_based_on_x_prev(
            x_t=x_t,
            x_prev=x0,
            x_pred_prev=x_t,
            mask=mask,
            keep_n_samples=conf["resample.keep_n_samples"],
        )[0]

        x_t = x_t.clamp(-1.0, 1.0)
        return {
            "sample": x_t,
        }


# implemenet
class O_DDIMSampler(DDIMSampler):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )
        assert conf.get("optimize_xt.optimize_xt",
                        False), "Double check on optimize"
        self.ddpm_num_steps = conf.get(
            "ddim.schedule_params.ddpm_num_steps", 250)
        self.coef_xt_reg = conf.get("optimize_xt.coef_xt_reg", 0.001)
        self.coef_xt_reg_decay = conf.get("optimize_xt.coef_xt_reg_decay", 1.0)
        self.num_iteration_optimize_xt = conf.get(
            "optimize_xt.num_iteration_optimize_xt", 1
        )
        self.lr_xt = conf.get("optimize_xt.lr_xt", 0.001)
        self.lr_xt_decay = conf.get("optimize_xt.lr_xt_decay", 1.0)
        self.use_smart_lr_xt_decay = conf.get(
            "optimize_xt.use_smart_lr_xt_decay", False
        )
        self.use_adaptive_lr_xt = conf.get(
            "optimize_xt.use_adaptive_lr_xt", False)
        self.mid_interval_num = int(conf.get("optimize_xt.mid_interval_num", 1))
        if conf.get("ddim.schedule_params.use_timetravel", False):
            self.steps = ddim_repaint_timesteps(**conf["ddim.schedule_params"])
        else:
            self.steps = ddim_timesteps(**conf["ddim.schedule_params"])
        
        self.mode = conf.get("mode", "inpaint")
        self.scale = conf.get("scale", 0)

    def p_sample(
        self,
        model_fn,
        x,
        t,
        prev_t,
        model_kwargs,
        lr_xt,
        coef_xt_reg,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):
        if self.mode == "inpaint":
            def loss_fn(_x0, _pred_x0, _mask):
                ret = torch.sum((_x0 * _mask - _pred_x0 * _mask) ** 2)
                return ret
        elif self.mode == "super_resolution":
            size = x.shape[-1]
            downop = nn.AdaptiveAvgPool2d(
                (size // self.scale, size // self.scale))

            def loss_fn(_x0, _pred_x0, _mask):
                down_x0 = downop(_x0)
                down_pred_x0 = downop(_pred_x0)
                ret = torch.sum((down_x0 - down_pred_x0) ** 2)
                return ret
        else:
            raise ValueError("Unkown mode: {self.mode}")

        def reg_fn(_origin_xt, _xt):
            ret = torch.sum((_origin_xt - _xt) ** 2)
            return ret

        def process_xstart(_x):
            if denoised_fn is not None:
                _x = denoised_fn(_x)
            if clip_denoised:
                return _x.clamp(-1, 1)
            return _x

        def get_et(_x, _t):
            if self.mid_interval_num > 1:
                res = grad_ckpt(
                    self._get_et, model_fn, _x, _t, model_kwargs, use_reentrant=False
                )
            else:
                res = self._get_et(model_fn, _x, _t, model_kwargs)
            return res

        def get_smart_lr_decay_rate(_t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num

            # 获取间隔为interval_num 的逆序数列表，头为当前t，尾为0
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)

            ret = 1
            time_pairs = list(zip(steps[:-1], steps[1:]))
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                ret *= self.sqrt_recip_alphas_cumprod[_cur_t] * math.sqrt(
                    self.alphas_cumprod[_prev_t]
                )
            return ret

        def multistep_predx0(_x, _et, _t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)
            time_pairs = list(zip(steps[:-1], steps[1:]))
            x_t = _x
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                _cur_t = torch.tensor([_cur_t] * _x.shape[0], device=_x.device)
                _prev_t = torch.tensor(
                    [_prev_t] * _x.shape[0], device=_x.device)
                if i != 0:
                    _et = get_et(x_t, _cur_t)
                x_t = grad_ckpt(
                    get_update, x_t, _cur_t, _prev_t, _et, None, use_reentrant=False
                )
            return x_t

        def get_predx0(_x, _t, _et, interval_num=1):
            if interval_num == 1:
                return process_xstart(self._predict_xstart_from_eps(_x, _t, _et))
            else:
                _pred_x0 = grad_ckpt(
                    multistep_predx0, _x, _et, _t, interval_num, use_reentrant=False
                )
                return process_xstart(_pred_x0)

        def get_update(  # x_t -> x_t-1
            _x,
            cur_t,
            _prev_t,
            _et=None,
            _pred_x0=None,
        ):
            if _et is None:
                _et = get_et(_x=_x, _t=cur_t)
            if _pred_x0 is None:
                _pred_x0 = get_predx0(_x, cur_t, _et, interval_num=1)

            alpha_t = _extract_into_tensor(self.alphas_cumprod, cur_t, _x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, _prev_t, _x.shape)
            sigmas = (   # 0
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )
            mean_pred = (
                _pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * _et  # dir_xt
            )
            noise = noise_like(_x.shape, _x.device, repeat=False)
            nonzero_mask = (cur_t != 0).float().view(-1,
                                                    *([1] * (len(_x.shape) - 1)))
            _x_prev = mean_pred + noise * sigmas * nonzero_mask
            return _x_prev

        B, C = x.shape[:2]
        assert t.shape == (B,)
        x0 = model_kwargs["gt"]
        mask = model_kwargs["gt_keep_mask"]

        # condition mean
        if cond_fn is not None:
            model_fn = self._wrap_model(model_fn)
            B, C = x.shape[:2]
            assert t.shape == (B,)
            model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            _, model_var_values = torch.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
            with torch.enable_grad():
                gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
                x = x + model_variance * gradient

        if self.use_smart_lr_xt_decay:
            lr_xt /= get_smart_lr_decay_rate(t, self.mid_interval_num)

        # optimize
        with torch.enable_grad():
            origin_x = x.clone().detach()
            x = x.detach().requires_grad_()
            e_t = get_et(_x=x, _t=t)  # 模型预测的噪音分布均值
            pred_x0 = get_predx0(     # 一步预测X0 
                _x=x, _t=t, _et=e_t, interval_num=self.mid_interval_num
            )
            prev_loss = loss_fn(x0, pred_x0, mask).item()  # X0和预测的X0（均值）的均方误差

            logging_info(f"step: {t[0].item()} lr_xt {lr_xt:.8f}")

            # 通过循环修正输入噪音Xt
            for step in range(self.num_iteration_optimize_xt):
                loss = loss_fn(x0, pred_x0, mask) + \
                    coef_xt_reg * reg_fn(origin_x, x)    # 公式10
                x_grad = torch.autograd.grad(
                    loss, x, retain_graph=False, create_graph=False
                )[0].detach()
                new_x = x - lr_xt * x_grad

                logging_info(
                    f"grad norm: {torch.norm(x_grad, p=2).item():.3f} "
                    f"{torch.norm(x_grad * mask, p=2).item():.3f} "
                    f"{torch.norm(x_grad * (1. - mask), p=2).item():.3f}"
                )

                while self.use_adaptive_lr_xt and True:  # 计算调整XT后的输入损失，如果值更大或NAN 则调整学习率
                    with torch.no_grad():
                        e_t = get_et(new_x, _t=t)
                        pred_x0 = get_predx0(
                            new_x, _t=t, _et=e_t, interval_num=self.mid_interval_num
                        )
                        new_loss = loss_fn(x0, pred_x0, mask) + coef_xt_reg * reg_fn(
                            origin_x, new_x
                        )
                        if not torch.isnan(new_loss) and new_loss <= loss:
                            break
                        else:
                            lr_xt *= 0.8
                            logging_info(
                                "Loss too large (%.3lf->%.3lf)! Learning rate decreased to %.5lf."
                                % (loss.item(), new_loss.item(), lr_xt)
                            )
                            del new_x, e_t, pred_x0, new_loss
                            new_x = x - lr_xt * x_grad

                # 调整输入的X 对于预测的噪音和X0预测
                x = new_x.detach().requires_grad_()
                e_t = get_et(x, _t=t)
                pred_x0 = get_predx0(
                    x, _t=t, _et=e_t, interval_num=self.mid_interval_num
                )
                del loss, x_grad
                torch.cuda.empty_cache()

        # after optimize
        with torch.no_grad():
            new_loss = loss_fn(x0, pred_x0, mask).item()
            logging_info("Loss Change: %.3lf -> %.3lf" % (prev_loss, new_loss))
            new_reg = reg_fn(origin_x, new_x).item()
            logging_info("Regularization Change: %.3lf -> %.3lf" % (0, new_reg))
            pred_x0, e_t, x = pred_x0.detach(), e_t.detach(), x.detach()
            del origin_x, prev_loss
            x_prev = get_update(
                x,
                t,
                prev_t,
                e_t,
                _pred_x0=pred_x0 if self.mid_interval_num == 1 else None,
            )

        return {"x": x, "x_prev": x_prev, "pred_x0": pred_x0, "loss": new_loss}

    def p_sample_loop(
        self,
        model_fn,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None,
        **kwargs,
    ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = torch.randn(shape, device=device)

        time_pairs = list(zip(self.steps[:-1], self.steps[1:]))

        x_t = img
        # set up hyper paramer for this run
        lr_xt = self.lr_xt  # 动态学习率系数超参数
        coef_xt_reg = self.coef_xt_reg  # 一致性惩罚项系数超参数
        loss = None

        status = None
        res = torch.from_numpy(self.alphas_cumprod).to(device=x_t.device)
        for cur_t, prev_t in tqdm(time_pairs):
            if cur_t > prev_t:  # denoise
                status = "reverse"
                    
                cur_t = torch.tensor([cur_t] * shape[0], device=device)
                prev_t = torch.tensor([prev_t] * shape[0], device=device)

                output = self.p_sample(
                    model_fn,
                    x=x_t,
                    t=cur_t,
                    prev_t=prev_t,
                    model_kwargs=model_kwargs,
                    pred_xstart=None,
                    lr_xt=lr_xt,
                    coef_xt_reg=coef_xt_reg,
                )
                x_t = output["x_prev"]
                loss = output["loss"]

                # lr decay
                if self.lr_xt_decay != 1.0:
                    logging_info(
                        "Learning rate of xt decay: %.5lf -> %.5lf."
                        % (lr_xt, lr_xt * self.lr_xt_decay)
                    )
                lr_xt *= self.lr_xt_decay
                if self.coef_xt_reg_decay != 1.0:
                    logging_info(
                        "Coefficient of regularization decay: %.5lf -> %.5lf."
                        % (coef_xt_reg, coef_xt_reg * self.coef_xt_reg_decay)
                    )
                coef_xt_reg *= self.coef_xt_reg_decay

            else:  # time travel back
                if status == "reverse" and conf.get(
                    "optimize_xt.optimize_before_time_travel", False
                ):
                    # update xt if previous status is reverse
                    x_t = self.get_updated_xt(
                        model_fn,
                        x=x_t,
                        t=torch.tensor([cur_t] * shape[0], device=device),
                        model_kwargs=model_kwargs,
                        lr_xt=lr_xt,
                        coef_xt_reg=coef_xt_reg,
                    )
                status = "forward"
                assert prev_t == cur_t + 1, "Only support 1-step time travel back"
                prev_t = torch.tensor([prev_t] * shape[0], device=device)
                with torch.no_grad():
                    x_t = self._undo(x_t, prev_t)
                # undo lr decay
                logging_info(f"Undo step: {cur_t}")
                lr_xt /= self.lr_xt_decay
                coef_xt_reg /= self.coef_xt_reg_decay

        x_t = x_t.clamp(-1.0, 1.0)  # normalize
        return {"sample": x_t, "loss": loss}

    def get_updated_xt(self, model_fn, x, t, model_kwargs, lr_xt, coef_xt_reg):
        return self.p_sample(
            model_fn,
            x=x,
            t=t,
            prev_t=torch.zeros_like(t, device=t.device),
            model_kwargs=model_kwargs,
            pred_xstart=None,
            lr_xt=lr_xt,
            coef_xt_reg=coef_xt_reg,
        )["x"]

class FJ_DDIMSampler(DDIMSampler):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        super().__init__(
            use_timesteps=use_timesteps,
            conf=conf,
            **kwargs,
        )
        assert conf.get("optimize_xt.optimize_xt",
                        False), "Double check on optimize"
        self.ddpm_num_steps = conf.get(
            "ddim.schedule_params.ddpm_num_steps", 250)
        self.coef_xt_reg = conf.get("optimize_xt.coef_xt_reg", 0.001)
        self.coef_xt_reg_decay = conf.get("optimize_xt.coef_xt_reg_decay", 1.0)
        self.num_iteration_optimize_xt = conf.get(
            "optimize_xt.num_iteration_optimize_xt", 1
        )
        self.lr_xt = conf.get("optimize_xt.lr_xt", 0.001)
        self.lr_xt_decay = conf.get("optimize_xt.lr_xt_decay", 1.0)
        self.use_smart_lr_xt_decay = conf.get(
            "optimize_xt.use_smart_lr_xt_decay", False
        )
        self.use_adaptive_lr_xt = conf.get(
            "optimize_xt.use_adaptive_lr_xt", False)
        self.mid_interval_num = int(conf.get("optimize_xt.mid_interval_num", 1))
        if conf.get("ddim.schedule_params.use_timetravel", False):
            self.steps = ddim_repaint_timesteps(**conf["ddim.schedule_params"])
        else:
            self.steps = ddim_timesteps(**conf["ddim.schedule_params"])
        
        self.mode = conf.get("mode", "inpaint")
        self.scale = conf.get("scale", 0)
        self.sqrt_alpcu = torch.from_numpy(self.sqrt_alphas_cumprod)
        self.accumulated_weight = None
        self.weight_t = None

        # 缓存配置参数
        self.crop_size = conf['crop']['image_size']
        self.gd_mask_radius = conf.gd_mask_radius
        self.multistage = conf["improve"]["multistage"]
        self.cropt_update_end_step = conf["improve"]["cropt_update_end_step"]
        self.cropt_change_step = conf["improve"]["cropt_change_step"]
        self.update_mask = conf["improve"]["update_mask"]


    def p_sample(
        self,
        model_fn,
        guide_models,
        x,
        t,
        prev_t,
        model_kwargs,
        lr_xt,
        coef_xt_reg,
        conf,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):

        def loss_fn(pre_x0_crop, pred_crop0, t, logg=None):
            '''
            输入为局部区域张量
            参数：
                pre_x0_crop: 生成的图像，取值范围为 [-1, 1]
                pred_crop0: 当前时间步的目标图像，取值范围为 [-1, 1]
                t: 当前时间步，张量类型
                logg: 日志记录器
            返回：
                total_loss: 总损失值，用于优化
            '''
            # Cache frequently accessed configuration
            improve_conf = conf.improve
            multiscale_loss = improve_conf.get("multiscale_loss")
            loss_weight_update_end = improve_conf.get("loss_weight_update_end")
            mask_save = improve_conf.get("mask_save")
            use_loss_weight = improve_conf.get("use_loss_weight")
            cropt_change_step = model_kwargs.get("cropt_change_step")

            # Extract current timestep
            current_step = t[0].item()

            # Determine and generate mask
            if use_loss_weight and self.multistage and current_step <= cropt_change_step and (current_step >= loss_weight_update_end) and (self.weight_t != current_step):
                # Update accumulated mask based on MSE
                compute_loss_weight(pre_x0_crop, pred_crop0)
                self.weight_t = current_step
                # Visualization: Save combined images every 3 steps if save_path is provided
                save_path = f"{conf.outdir}/fj_ddim/masks/" if mask_save else None
                if save_path:
                    save_combined_images(pre_x0_crop, pred_crop0, self.accumulated_weight, save_path, step)

            if multiscale_loss:
                al_cumd_t = self.sqrt_alpcu[current_step]
                # Compute Laplacian pyramid loss
                lapl_l = dynamic_laplacian_loss(
                    pre_x0_crop,
                    pred_crop0,
                    al_cumd_t,
                    mask=self.accumulated_weight,
                )
                # Compute the original loss with the accumulated mask
                original_l = original_loss(pre_x0_crop, pred_crop0, self.accumulated_weight)

                # Calculate loss weights
                lambda_original = al_cumd_t
                lambda_laplacian = 100.0 * (1.0 - lambda_original)

                # Combine losses to get the total loss
                total_loss = lambda_laplacian * lapl_l + lambda_original * original_l

                # Logging detailed loss information
                if logg:
                    log_msg = (
                        f"total_loss: {total_loss.item():.4f}; "
                        f"laplacian_loss: {lapl_l.item():.4f} * {lambda_laplacian:.2f}; "
                        f"original_loss: {original_l.item():.4f} * {lambda_original:.2f}"
                    )
                    logging_info(log_msg)
            else:
                # Compute original loss without multiscale components
                total_loss = original_loss(pre_x0_crop, pred_crop0, self.accumulated_weight)

                # Logging total loss
                if logg:
                    log_msg = f"original_loss: {total_loss.item():.4f}"
                    logging_info(log_msg)

            return total_loss

        def compute_loss_weight(_x0, _pred_x0, gamma=1.0, bias=0.0):
            """
            基于均方差生成单通道显著性蒙版
            """
            # 计算均方差差异
            diff = (_x0 - _pred_x0) ** 2  # (N, C, H, W)
        
            # 限制范围
            # diff = diff.amax(dim=1, keepdim=True) 
            diff = diff.mean(dim=1, keepdim=True)  # 求最大值 (batch_size, 1, 64, 64)
            diff = diff ** gamma
            # diff_min = th.amin(diff, dim=( 2, 3), keepdim=True)  # (batch_size, 1, 1, 1)
            diff_max = torch.amax(diff, dim=( 2, 3), keepdim=True)  # (batch_size, 1, 1, 1)
            upper = diff_max * 0.2
            new_mask = diff / (upper + 1e-8)
            new_mask = (new_mask + bias).clamp(0, 1)

            # 处理累积蒙版
            if self.accumulated_weight is None:
                self.accumulated_weight = new_mask.clone()
            else:
                delta_mask = (new_mask - self.accumulated_weight).clamp(0, 1)
                self.accumulated_weight = (self.accumulated_weight + delta_mask)
                self.accumulated_weight = self.accumulated_weight.clamp(0, 1)

        
        def reg_fn(_origin_xt, _xt):
            ret = torch.sum((_origin_xt - _xt) ** 2)
            return ret

        def process_xstart(_x):
            if denoised_fn is not None:
                _x = denoised_fn(_x)
            if clip_denoised:
                return _x.clamp(-1, 1)
            return _x

        def get_et(_x, _t):
            if self.mid_interval_num > 1:
                res = grad_ckpt(
                    self._get_et, model_fn, _x, _t, model_kwargs, use_reentrant=False
                )
            else:
                res = self._get_et(model_fn, _x, _t, model_kwargs)
            return res
    
        def get_smart_lr_decay_rate(_t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num

            # 获取间隔为interval_num 的逆序数列表，头为当前t，尾为0
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)

            ret = 1
            time_pairs = list(zip(steps[:-1], steps[1:]))
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                ret *= self.sqrt_recip_alphas_cumprod[_cur_t] * math.sqrt(
                    self.alphas_cumprod[_prev_t]
                )
            return ret

        def multistep_predx0(_x, _et, _t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)
            time_pairs = list(zip(steps[:-1], steps[1:]))
            x_t = _x
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                _cur_t = torch.tensor([_cur_t] * _x.shape[0], device=_x.device)
                _prev_t = torch.tensor(
                    [_prev_t] * _x.shape[0], device=_x.device)
                if i != 0:
                    _et = get_et(x_t, _cur_t)
                x_t = grad_ckpt(
                    get_update, x_t, _cur_t, _prev_t, _et, None, use_reentrant=False
                )
            return x_t

        def get_predx0(_x, _t, _et, interval_num=1):
            if interval_num == 1:
                return process_xstart(self._predict_xstart_from_eps(_x, _t, _et))
            else:
                _pred_x0 = grad_ckpt(
                    multistep_predx0, _x, _et, _t, interval_num, use_reentrant=False
                )
                return process_xstart(_pred_x0)
        
        def get_update(  # x_t -> x_t-1
            _x,
            cur_t,
            _prev_t,
            _et=None,
            _pred_x0=None,
        ):
            if _et is None:
                _et = get_et(_x=_x, _t=cur_t)
            if _pred_x0 is None:
                _pred_x0 = get_predx0(_x, cur_t, _et, interval_num=1)

            alpha_t = _extract_into_tensor(self.alphas_cumprod, cur_t, _x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, _prev_t, _x.shape)
            sigmas = (   # 0
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )
            mean_pred = (
                _pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * _et  # dir_xt
            )
            noise = noise_like(_x.shape, _x.device, repeat=False)
            nonzero_mask = (cur_t != 0).float().view(-1,
                                                    *([1] * (len(_x.shape) - 1)))
            _x_prev = mean_pred + noise * sigmas * nonzero_mask
            return _x_prev

        B, C = x.shape[:2]
        assert t.shape == (B,)
        crop = model_kwargs["crop_t"]
        mask = model_kwargs["gt_keep_mask"]
        location = model_kwargs["location"] # list[x, y]
        gd_mask = model_kwargs["gd_mask"]
        crop_gd_mask = model_kwargs["crop_gd_mask"]
        local_labels = model_kwargs["crop_y"]
        crop_model_fn = guide_models
        cropt_update_end_step = model_kwargs["cropt_update_end_step"]
        cropt_change_step = model_kwargs["cropt_change_step"]

        # condition mean
        if cond_fn is not None:
            model_fn = self._wrap_model(model_fn)
            B, C = x.shape[:2]
            assert t.shape == (B,)
            model_output = model_fn(x, self._scale_timesteps(t), **model_kwargs)
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            _, model_var_values = torch.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
            with torch.enable_grad():
                gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
                x = x + model_variance * gradient

        # lr
        if self.use_smart_lr_xt_decay:
            ori_lr = lr_xt
            ori_lr /= get_smart_lr_decay_rate(t, self.mid_interval_num)
            step = t[0]
            if self.multistage:
                if step > cropt_update_end_step:
                    lr_xt, lr_cropt = ori_lr, 0.0
                elif step > cropt_change_step:
                    lr_xt, lr_cropt = 0.0, 0.0
                elif self.update_mask:
                    lr_xt, lr_cropt = ori_lr, ori_lr
                else:
                    lr_xt, lr_cropt = ori_lr, 0.0
            else:
                lr_xt, lr_cropt = ori_lr, 0.0

        # Optimized Code
        if lr_xt == 0.0 and lr_cropt == 0.0:
            step_num = t[0].item()
            logging_info(f"step: {step_num} lr equal 0, skip optimize... ")
            
            with torch.no_grad():
                # Compute predictions without optimization
                e_t = get_et(_x=x, _t=t)  # Model predicted noise mean
                pred_x0 = get_predx0(_x=x, _t=t, _et=e_t, interval_num=self.mid_interval_num)
                crop_e_t = self._get_et(
                    crop_model_fn, crop, t, model_kwargs={"y": model_kwargs["crop_y"]}
                )
                pred_crop0 = get_predx0(_x=crop, _t=t, _et=crop_e_t)
                new_loss = 0.0
        else:
            # Optimize
            with torch.enable_grad():
                # Clone and detach inputs once to preserve original states
                origin_x = x.detach().clone()
                origin_crop = crop.detach().clone()

                # Enable gradients for optimization
                x = origin_x.detach().requires_grad_(True)
                crop = origin_crop.detach().requires_grad_(True)

                # Precompute crop indices to avoid recalculating in loops
                crop_y_start, crop_x_start = location[1], location[0]
                crop_size = conf['crop']['image_size']
                crop_y_end = crop_y_start + crop_size
                crop_x_end = crop_x_start + crop_size

                # Perform initial predictions for the large image
                e_t = get_et(_x=x, _t=t)  # Model predicted noise mean
                pred_x0 = get_predx0(
                    _x=x, _t=t, _et=e_t, interval_num=self.mid_interval_num
                )
                pred_x0_crop = pred_x0[:, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

                # Perform initial predictions for the small image (crop)
                crop_e_t = self._get_et(
                    crop_model_fn, crop, t, model_kwargs={"y": model_kwargs["crop_y"]}
                )
                pred_crop0 = get_predx0(_x=crop, _t=t, _et=crop_e_t)

                # Compute initial loss between large and small image predictions
                prev_loss = loss_fn(pred_x0_crop, pred_crop0, t).item()

                # Logging current step and learning rates
                step_num = t[0].item()
                logging_info(f"step: {step_num}")
                logging_info(f"lr_xt {lr_xt:.8f} lr_cropt {lr_cropt:.8f}")

                # Retrieve the number of gradient steps for the current timestep
                step_count = get_g_scha(length=self.num_timesteps, **conf.g_scha)[t[0]]

                # Precompute regularization coefficient
                coef_reg = coef_xt_reg

                # Precompute static parts of the regularization term
                reg_fn_origin_x = lambda x_new: reg_fn(origin_x, x_new)
                reg_fn_origin_crop = lambda crop_new: reg_fn(origin_crop, crop_new)

                # Iterate over the number of gradient steps
                for _ in range(step_count):
                    # Compute loss with regularization
                    reg_term = reg_fn_origin_x(x) + reg_fn_origin_crop(crop)
                    loss = loss_fn(pred_x0_crop, pred_crop0, t, logg=True) + coef_reg * reg_term

                    # Compute gradients with respect to x and crop
                    x_grad, crop_grad = torch.autograd.grad(
                        outputs=loss,
                        inputs=[x, crop],
                        retain_graph=False,
                        create_graph=False,
                    )

                    # Update x and crop using gradient descent
                    new_x = x - lr_xt * gd_mask * x_grad.detach()
                    new_crop = crop - lr_cropt * crop_gd_mask * crop_grad.detach()

                    # Adaptive learning rate adjustment if enabled
                    if self.use_adaptive_lr_xt:
                        while True:
                            with torch.no_grad():
                                # Compute predictions with updated x and crop
                                e_t_new = get_et(_x=new_x, _t=t)
                                pred_x0_new = get_predx0(
                                    _x=new_x, _t=t, _et=e_t_new, interval_num=self.mid_interval_num
                                )
                                pred_x0_crop_new = pred_x0_new[
                                    :, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end
                                ]

                                crop_e_t_new = self._get_et(
                                    crop_model_fn, new_crop, t, model_kwargs={"y": model_kwargs["crop_y"]}
                                )
                                pred_crop0_new = get_predx0(_x=new_crop, _t=t, _et=crop_e_t_new)

                                # Compute new loss with updated predictions
                                reg_new = reg_fn_origin_x(new_x) + reg_fn_origin_crop(new_crop)
                                new_loss = (
                                    loss_fn(pred_x0_crop_new, pred_crop0_new, t).item()
                                    + coef_reg * reg_new
                                )

                                # Check if the new loss is acceptable
                                if not math.isnan(new_loss) and new_loss <= loss.item():
                                    break  # Accept the update
                                else:
                                    # Reduce learning rates by a factor of 0.8
                                    lr_xt *= 0.8
                                    lr_cropt *= 0.8
                                    logging_info(
                                        f"Loss too large ({loss.item():.3f}->{new_loss:.3f})! "
                                        f"Learning rate decreased to {lr_xt:.5f}."
                                    )
                                    # Recompute updates with the reduced learning rates
                                    new_x = x - lr_xt * gd_mask * x_grad.detach()
                                    new_crop = crop - lr_cropt * crop_gd_mask * crop_grad.detach()

                    # Update x and crop for the next iteration, enabling gradients
                    x = new_x.detach().requires_grad_(True)
                    crop = new_crop.detach().requires_grad_(True)

                    # Recompute predictions with the updated x and crop
                    e_t = get_et(_x=x, _t=t)
                    pred_x0 = get_predx0(
                        _x=x, _t=t, _et=e_t, interval_num=self.mid_interval_num
                    )
                    pred_x0_crop = pred_x0[:, :, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

                    crop_e_t = self._get_et(
                        crop_model_fn, crop, t, model_kwargs={"y": model_kwargs["crop_y"]}
                    )
                    pred_crop0 = get_predx0(_x=crop, _t=t, _et=crop_e_t)

                    # Clean up to free memory
                    del loss, x_grad, crop_grad
                    torch.cuda.empty_cache()

            # After optimization
            with torch.no_grad():
                if lr_xt != 0.0 or lr_cropt != 0.0:
                    new_loss = loss_fn(pred_x0_crop, pred_crop0, t).item()
                    logging_info(f"After update, Loss Change: {prev_loss:.3f} -> {new_loss:.3f}")

                    if 'new_x' in locals() and new_x is not None:
                        new_reg_x = reg_fn(origin_x, new_x).item()
                        logging_info(f"X Regularization Change: 0.000 -> {new_reg_x:.3f}")
                    if 'new_crop' in locals() and new_crop is not None:
                        new_reg_crop = reg_fn(origin_crop, new_crop).item()
                        logging_info(f"Crop Regularization Change: 0.000 -> {new_reg_crop:.3f}")

                    # Detach tensors to prevent gradient tracking
                    pred_x0 = pred_x0.detach()
                    e_t = e_t.detach()
                    x = x.detach()
                    pred_crop0 = pred_crop0.detach()
                    crop_e_t = crop_e_t.detach()
                    crop = crop.detach()

                    # Clean up to free memory
                    del origin_x, origin_crop, prev_loss

        # Update previous states
        x_prev = get_update(
            x,
            t,
            prev_t,
            e_t,
            _pred_x0=pred_x0 if self.mid_interval_num == 1 else None,
        )
        crop_prev = get_update(
            crop,
            t,
            prev_t,
            crop_e_t,
            _pred_x0=pred_crop0 if self.mid_interval_num == 1 else None,
        )
        del e_t, crop_e_t
        return {
            "x": x, 
            "x_prev": x_prev, 
            "pred_x0": pred_x0, 
            "crop": crop,
            "crop_prev": crop_prev,
            "pred_crop0": pred_crop0,
            "loss": new_loss, 
            }
    
    def p_sample_ori_ddim(
        self,
        model_fn,
        x,
        t,
        prev_t,
        model_kwargs,
        clip_denoised=True,
        denoised_fn=None,
        **kwargs,
    ):

        def process_xstart(_x):
            if denoised_fn is not None:
                _x = denoised_fn(_x)
            if clip_denoised:
                return _x.clamp(-1, 1)
            return _x

        def get_et(_x, _t):
            if self.mid_interval_num > 1:
                res = grad_ckpt(
                    self._get_et, model_fn, _x, _t, model_kwargs, use_reentrant=False
                )
            else:
                res = self._get_et(model_fn, _x, _t, model_kwargs)
            return res

        def multistep_predx0(_x, _et, _t, interval_num):
            int_t = int(_t[0].item())
            interval = int_t // interval_num
            steps = (
                (np.arange(0, interval_num) * interval)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
            steps = steps.tolist()
            if steps[0] != int_t:
                steps.insert(0, int_t)
            if steps[-1] != 0:
                steps.append(0)
            time_pairs = list(zip(steps[:-1], steps[1:]))
            x_t = _x
            for i in range(len(time_pairs)):
                _cur_t, _prev_t = time_pairs[i]
                _cur_t = torch.tensor([_cur_t] * _x.shape[0], device=_x.device)
                _prev_t = torch.tensor(
                    [_prev_t] * _x.shape[0], device=_x.device)
                if i != 0:
                    _et = get_et(x_t, _cur_t)
                x_t = grad_ckpt(
                    get_update, x_t, _cur_t, _prev_t, _et, None, use_reentrant=False
                )
            return x_t

        def get_predx0(_x, _t, _et, interval_num=1):
            if interval_num == 1:
                return process_xstart(self._predict_xstart_from_eps(_x, _t, _et))
            else:
                _pred_x0 = grad_ckpt(
                    multistep_predx0, _x, _et, _t, interval_num, use_reentrant=False
                )
                return process_xstart(_pred_x0)
        
        def get_update(  # x_t -> x_t-1
            _x,
            cur_t,
            _prev_t,
            _et=None,
            _pred_x0=None,
        ):
            if _et is None:
                _et = get_et(_x=_x, _t=cur_t)
            if _pred_x0 is None:
                _pred_x0 = get_predx0(_x, cur_t, _et, interval_num=1)

            alpha_t = _extract_into_tensor(self.alphas_cumprod, cur_t, _x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, _prev_t, _x.shape)
            sigmas = (   # 0
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )
            mean_pred = (
                _pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * _et  # dir_xt
            )
            noise = noise_like(_x.shape, _x.device, repeat=False)
            nonzero_mask = (cur_t != 0).float().view(-1,
                                                    *([1] * (len(_x.shape) - 1)))
            _x_prev = mean_pred + noise * sigmas * nonzero_mask
            return _x_prev

        B, C = x.shape[:2]
        assert t.shape == (B,)

        # after optimize
        with torch.no_grad():
            e_t = get_et(_x=x, _t=t)  # 模型预测的噪音分布均值
            pred_x0 = get_predx0(_x=x, _t=t, _et=e_t, interval_num=self.mid_interval_num)
            x_prev = get_update(
                x,
                t,
                prev_t,
                e_t,
                _pred_x0=pred_x0 if self.mid_interval_num == 1 else None,
            )

        return {
            "x": x, 
            "x_prev": x_prev, 
            "pred_x0": pred_x0, 
            }
    
    def p_sample_simple(
        self,
        model_fn,
        x,
        t,
        prev_t,
        model_kwargs,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        **kwargs,
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        with torch.no_grad():
            alpha_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            alpha_prev = _extract_into_tensor(
                self.alphas_cumprod, prev_t, x.shape)
            sigmas = (
                self.ddim_sigma
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt((1 - alpha_t / alpha_prev))
            )

            def process_xstart(_x):
                if denoised_fn is not None:
                    _x = denoised_fn(_x)
                if clip_denoised:
                    return _x.clamp(-1, 1)
                return _x

            e_t = self._get_et(model_fn, x, t, model_kwargs)
            pred_x0 = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=e_t))

            mean_pred = (
                pred_x0 * torch.sqrt(alpha_prev)
                + torch.sqrt(1 - alpha_prev - sigmas**2) * e_t
            )
            noise = noise_like(x.shape, x.device, repeat=False)

            nonzero_mask = (t != 0).float().view(-1, *
                                                ([1] * (len(x.shape) - 1)))
            x_prev = mean_pred + noise * sigmas * nonzero_mask

        return {
            "x_prev": x_prev,
            "pred_x0": pred_x0,
        }
        
    def p_sample_loop(
            self,
            model_fn,
            shape,
            guide_models,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
            return_all=False,
            conf=None,
            **kwargs,
        ):
        if device is None:
            device = next(model_fn.parameters()).device
        assert isinstance(shape, (tuple, list)), "Shape must be a tuple or list"
        
        # 解包位置坐标
        loc_x, loc_y = model_kwargs["location"]  # location 为 [x, y]
        
        # 使用类属性
        crop_size = self.crop_size
        gd_mask_radius = self.gd_mask_radius
        multistage = self.multistage
        cropt_update_end_step = self.cropt_update_end_step
        cropt_change_step = self.cropt_change_step
        update_mask = self.update_mask
        
        # 生成初始噪音和裁剪区域
        img = torch.randn(shape, device=device)
        crop = img[:, :, loc_y:loc_y + crop_size, loc_x:loc_x + crop_size].clone()
        
        # 设置超参数
        lr_xt = self.lr_xt  # 动态学习率系数
        coef_xt_reg = self.coef_xt_reg  # 一致性惩罚项系数
        
        # 初始化变量
        loss = None
        x_t = img.clone()
        crop_t = crop.clone()
        
        # 创建梯度下降掩码函数
        cx = loc_x + crop_size // 2
        cy = loc_y + crop_size // 2
        gd_mask_fn = partial(
            create_custom_gaussian_mask,
            radius=gd_mask_radius,
            channels=shape[1],
            batch_size=shape[0],
            device=device
        )
        
        # 初始化 gt_keep_mask
        gt_keep_mask = torch.zeros_like(x_t)
        gt_keep_mask[:, :, loc_y:loc_y + crop_size, loc_x:loc_x + crop_size] = 1.0
        model_kwargs["gt_keep_mask"] = gt_keep_mask
        model_kwargs["crop_t"] = crop_t
        
        # 生成
        if update_mask:
            gd_mask = gd_mask_fn(center=(cx, cy), size=shape[-1], falloff_factor=1e-6, mode="mask1")
            crop_gd_mask = gd_mask_fn(center=(crop_size // 2, crop_size // 2), size=crop_size, falloff_factor=1e-6, mode="mask2")
        else:
            gd_mask = torch.ones_like(x_t)
            crop_gd_mask = torch.ones_like(crop_t)

        model_kwargs["gd_mask"] = gd_mask
        model_kwargs["crop_gd_mask"] = crop_gd_mask
        
        # 多阶段设置
        model_kwargs["cropt_update_end_step"] = cropt_update_end_step
        model_kwargs["cropt_change_step"] = cropt_change_step
        
        # 初始化掩码和状态
        self.accumulated_weight = None
        self.weight_t = None
        status = None
        x_t_process = []
        crop_process = []
        progress_t = 0
        c_t_first = True
        
        # 生成时间步对
        time_pairs = list(zip(self.steps[:-1], self.steps[1:]))
        loop = tqdm(time_pairs) if progress else time_pairs
        
        # 预计算裁剪区域的结束坐标
        crop_end_x = loc_x + crop_size
        crop_end_y = loc_y + crop_size
        
        for cur_t, prev_t in loop:
            # 多阶段：在特定步数替换裁剪图像
            if multistage and cur_t == cropt_change_step and c_t_first:
                crop_t = x_t[:, :, loc_y:crop_end_y, loc_x:crop_end_x].clone()
                logging_info(f"crop_t clip from xt at step {cur_t}...")
                c_t_first = False
            
            if cur_t > prev_t:  # 去噪阶段
                status = "reverse"
                model_kwargs["crop_t"] = crop_t
                
                # 每5步更新一次梯度掩码
                if prev_t % 5 == 0 and update_mask:
                    factor = float(self.sqrt_alpcu[prev_t]**2)
                    gd_mask = gd_mask_fn(center=(cx, cy), size=shape[-1], falloff_factor=factor, mode="mask1")
                    crop_gd_mask = gd_mask_fn(center=(crop_size // 2, crop_size // 2), size=crop_size, falloff_factor=factor, mode="mask2")
                    model_kwargs["gd_mask"] = gd_mask
                    model_kwargs["crop_gd_mask"] = crop_gd_mask
                    logging_info(f"Gd mask update, Step:{prev_t} , Falloff_factor: {factor:.8f}")
                
                # 创建时间步张量
                t_tensor = torch.full((shape[0],), cur_t, device=device, dtype=torch.long)
                prev_t_tensor = torch.full((shape[0],), prev_t, device=device, dtype=torch.long)
                
                # 执行采样步骤
                output = self.p_sample(
                    model_fn=model_fn,
                    guide_models=guide_models,
                    x=x_t,
                    t=t_tensor,
                    prev_t=prev_t_tensor,
                    model_kwargs=model_kwargs,
                    pred_xstart=None,
                    lr_xt=lr_xt,
                    coef_xt_reg=coef_xt_reg,
                    conf=conf,
                )
                x_t = output.get("x_prev")
                loss = output.get("loss")
                crop_t = output.get("crop_prev")
                
                # 更新过程列表
                if prev_t % 50 == 0:
                    if prev_t == progress_t:
                        x_t_process[-1] = output.get("pred_x0")
                        crop_process[-1] = output.get("pred_crop0")
                    else:
                        x_t_process.append(output.get("pred_x0"))
                        crop_process.append(output.get("pred_crop0"))
                        progress_t = prev_t
                
                # 学习率衰减
                if self.lr_xt_decay != 1.0:
                    logging_info(f"Learning rate of xt decay: {lr_xt:.5f} -> {lr_xt * self.lr_xt_decay:.5f}.")
                lr_xt *= self.lr_xt_decay
                if self.coef_xt_reg_decay != 1.0:
                    logging_info(f"Coefficient of regularization decay: {coef_xt_reg:.5f} -> {coef_xt_reg * self.coef_xt_reg_decay:.5f}.")
                coef_xt_reg *= self.coef_xt_reg_decay
            
            else:  # 时间旅行回退阶段
                if status == "reverse" and conf.get("optimize_xt.optimize_before_time_travel"):
                    # 如果之前状态为 reverse，则更新 x_t 和 crop_t
                    model_kwargs["crop_t"] = crop_t
                    x_t, crop_t = self.get_updated_xt(
                        model_fn=model_fn,
                        guide_models=guide_models,
                        x=x_t,
                        t=torch.full((shape[0],), cur_t, device=device, dtype=torch.long),
                        model_kwargs=model_kwargs,
                        lr_xt=lr_xt,
                        coef_xt_reg=coef_xt_reg,
                        conf=conf,                        
                    )
                status = "forward"
                assert prev_t == cur_t + 1, "Only support 1-step time travel back"
                
                # 创建前一步时间步张量
                prev_t_tensor = torch.full((shape[0],), prev_t, device=device, dtype=torch.long)
                with torch.no_grad():
                    x_t = self._undo(x_t, prev_t_tensor)
                    crop_t = self._undo(crop_t, prev_t_tensor)
                
                # 恢复学习率
                logging_info(f"Undo step: {cur_t}")
                lr_xt /= self.lr_xt_decay
                coef_xt_reg /= self.coef_xt_reg_decay
        
        # 构建目标张量 gt
        gt = torch.zeros_like(x_t)
        gt[:, :, loc_y:crop_end_y, loc_x:crop_end_x] = crop_t
        
        return {
            "sample": x_t, 
            "gt": gt, 
            "loss": loss, 
            "crop_process": crop_process, 
            "x_t_process": x_t_process, 
            "crop": crop_t,
        }

    
    def get_updated_xt(self, model_fn, guide_models, x, t, model_kwargs, lr_xt, coef_xt_reg, conf):
        result = self.p_sample(
            model_fn,
            guide_models,
            x=x,
            t=t,
            prev_t=torch.zeros_like(t, device=t.device),
            model_kwargs=model_kwargs,
            pred_xstart=None,
            lr_xt=lr_xt,
            coef_xt_reg=coef_xt_reg,
            conf=conf,
        )
        return result["x"], result["crop"]
    

    
