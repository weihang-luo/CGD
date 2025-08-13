# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import json
import os
import argparse
import random
import numpy as np
import torch as th
import torch.nn.functional as F
import conf_mgt
from utils import yamlread
from pathlib import Path
import torch.nn as nn

from utils import save_grid, save_image, normalize_image
from utils.config import Config
from utils.logger import get_logger, logging_info
from utils.nn_utils import get_all_paths, set_random_seed
from utils.result_recorder import ResultRecorder
from utils.timer import Timer

from guided_diffusion.respace import SpacedDiffusion
from guided_diffusion import (
    DDIMSampler,
    O_DDIMSampler,
    FJ_DDIMSampler,
)

from guided_diffusion.script_util import (
    model_defaults,
    create_model,
    diffusion_defaults,
    create_gaussian_diffusion,
    select_args,
)


class DifferenceConvNet(nn.Module):
    def __init__(self, kernel_size=3):
        super(DifferenceConvNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        # Initialize the kernel with ones for averaging
        with th.no_grad():
            self.conv.weight.fill_(1.0 / (kernel_size * kernel_size))

    def forward(self, x):
        return self.conv(x)

def generate_difference_mask(tensor1, tensor2, kernel_size, alpha=0.7, beta=0.5, device=None):
    # Ensure the tensors have the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("The input tensors must have the same shape")

    # Compute the absolute difference
    diff = th.abs(tensor1 - tensor2)

    # Convert to grayscale by averaging across the channel dimension
    diff_gray = diff.mean(dim=1, keepdim=True)

    # Use convolution to compute local differences
    model = DifferenceConvNet(kernel_size=kernel_size).to(device)
    pooled_diff = model(diff_gray)

    # Dynamically determine the threshold
    threshold = alpha * pooled_diff.mean() + beta * pooled_diff.std()

    logging_info(f"generate_difference_mask:" \
                 f"[alpha]={alpha:.1f}, [beta]={beta:.1f}, " \
                 f"[mean]={pooled_diff.mean():.4f}, [std]={pooled_diff.std():.4f}, [threshold]={threshold:.4f}...")

    # Apply the threshold to the pooled difference to generate the mask
    mask = (pooled_diff <= threshold).float().repeat(1, 3, 1, 1)

    return mask

def prepare_model(algorithm, conf, num_class, device):
    logging_info("Prepare model...")
    unet = create_model(**select_args(conf, model_defaults().keys()),)
    crop_model = create_model(
            **conf['crop'], num_classes=num_class,
        )
    
    SAMPLER_CLS = {
        "repaint": SpacedDiffusion,
        "ddim": DDIMSampler,
        "o_ddim": O_DDIMSampler,
        "fj_ddim": FJ_DDIMSampler,
    }
    sampler_cls = SAMPLER_CLS[algorithm]

    logging_info(f'create gaussian_diffusion{[sampler_cls.__name__]}...')
    sampler = create_gaussian_diffusion(
        **select_args(conf, diffusion_defaults().keys()),
        conf=conf,
        base_cls=sampler_cls,
    )

    logging_info(f"Loading model from {conf.model_path}...")
    unet.load_state_dict(th.load(conf.model_path, weights_only=True))
    logging_info(f"Loading crop from {conf.crop_model_path}...")
    crop_model.load_state_dict(th.load(conf.crop_model_path, weights_only=True))
    unet.to(device)
    crop_model.to(device)
    
    if conf.use_fp16:
        unet.convert_to_fp16()
        crop_model.convert_to_fp16()
    unet.eval()
    crop_model.eval()

    logging_info(f'create gaussian_diffusion{[sampler_cls.__name__]}...')
    sampler = create_gaussian_diffusion(
        **select_args(conf, diffusion_defaults().keys()),
        conf=conf,
        base_cls=sampler_cls,
    )

    return unet, sampler, crop_model

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample

def draw_boxes_on_batch(images, boxes, color=(1.0, -1.0, -1.0), thickness=2):
    B, C, H, W = images.shape
    color_tensor = th.tensor(color).view(C, 1, 1).to(images.device)  # Color tensor with shape (C, 1, 1)

    x, y, side_length = boxes
    
    # Draw top and bottom borders
    images[:, :, y:y+thickness, x:x+side_length] = color_tensor
    images[:, :, y+side_length-thickness:y+side_length, x:x+side_length] = color_tensor
    
    # Draw left and right borders
    images[:, :, y:y+side_length, x:x+thickness] = color_tensor
    images[:, :, y:y+side_length, x+side_length-thickness:x+side_length] = color_tensor
        
    return images

def main(conf: conf_mgt.Default_Conf):
    ###################################################################################
    # prepare conf, logger and recorder
    ###################################################################################
    all_paths = get_all_paths(os.path.join(conf.outdir, conf.algorithm))
    conf.update(all_paths)
    get_logger(all_paths["path_log"], force_add_handler=True)
    recorder = ResultRecorder(
        path_record=all_paths["path_record"],
        initial_record=conf,
        use_git=conf.use_git,
    )
    
    # 设置随机种子以确保可复现性
    if conf.seed is not None:
        logging_info(f"Setting random seed to {conf.seed} for reproducibility")
        set_random_seed(conf.seed, deterministic=True, no_torch=False, no_tf=True)
        # 设置环境变量以确保 CUDA 操作的确定性
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        th.use_deterministic_algorithms(True, warn_only=True)
        logging_info("Deterministic algorithms enabled for reproducibility")
    else:
        logging_info("No seed specified, results may not be reproducible")

    config = Config(default_config_dict=conf, use_argparse=False)

    # 局部分类
    if conf.crop_class_json_path:
        with Path(conf.crop_class_json_path).open('r') as f:
            defect_class = json.load(f)
    else:
        defect_class = {}


    ###################################################################################
    # prepare model and device
    ###################################################################################

    device = th.device(f"cuda:{conf.cuda}")  # if th.cuda.is_available() else th.device("cpu")
    unet, sampler, guide_models = prepare_model(conf.algorithm, config, len(defect_class), device)


    def model_fn(x, t, y=None, gt=None, **kwargs):
        return unet(x, t, y if conf.class_cond else None, gt=gt)
    
    cond_fn = None

    ###################################################################################
    # start sampling
    ###################################################################################
    logging_info(f"Start sampling,{conf.schedule_jump_params},final_t:{conf.final_t}")
    timer, num_image = Timer(), 0
    batch_size = conf.n_samples

    max_iterations = conf.max_iter
    iteration = 0
    
    # 为可复现性创建独立的随机数生成器
    if conf.seed is not None:
        rng = np.random.RandomState(conf.seed)
        logging_info(f"Using seeded random generator for reproducible location sampling")
    else:
        rng = np.random
        logging_info("Using default random generator (results may vary)")
    
    while iteration < max_iterations:
        iteration += 1
        logging_info(f"生成第：{iteration}/{max_iterations} 批次{batch_size}张图像...")
        model_kwargs = {}

        class_ind = 2 

        defect_y = th.full((batch_size,), class_ind, device=device)
        

        lx = rng.randint(20, conf.image_size - conf["classifier_image_size"]-20)
        ly = rng.randint(20, conf.image_size - conf["classifier_image_size"]-20)
        logging_info(f"Generated location: x={lx}, y={ly}")

        image_name =  f"{iteration:04}_{list(defect_class.keys())[class_ind]}_x{lx}y{ly}r{conf.gd_mask_radius}"
        
        model_kwargs["crop_y"] = defect_y
        model_kwargs["location"] = [lx, ly]

        # prepare save dir
        sample_path = all_paths["path_sample"]
        gird_path = all_paths["path_gird"]
        crop_path = all_paths["path_crop"]
        os.makedirs(sample_path, exist_ok=True)
        os.makedirs(gird_path, exist_ok=True)
        os.makedirs(crop_path, exist_ok=True)

        shape = (batch_size, 3, conf.image_size, conf.image_size)
        # sample images
        samples = []
        timer.start()
        result = sampler.p_sample_loop(
            model_fn,
            shape=shape,
            guide_models=guide_models,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            conf=config,
        )
        timer.end()
        ###################################################################################
        # save
        ###################################################################################
        
        if "loss" in result.keys() and result["loss"] is not None:
            recorder.add_with_logging(
                key=f"Final loss： {image_name}", value=result["loss"])

        # patch
        gt = normalize_image(result["gt"].detach().cpu())
        samples.append(gt)

        pred_x0_process = [normalize_image(ts).detach().cpu() for ts in result["x_t_process"]]
    
        samples = samples + pred_x0_process

        inpainted = normalize_image(result["sample"])
        crops = normalize_image(result["crop"])
        base_count = 1
        for sample, crop in zip(inpainted, crops):
            save_image(sample, os.path.join(sample_path, f"{image_name}_{base_count:04}.png"))
            save_image(crop, os.path.join(crop_path, f"{image_name}_{base_count:04}_crop.png"))
            base_count += 1

        samples = th.cat(samples)
        save_grid(
            samples,
            os.path.join(gird_path, f"{image_name}_grid.png"),
            nrow=batch_size,
        )


        # pred_crop0_process = normalize_image(th.cat(result["crop_process"])).detach().cpu()
        # save_grid(
        #     pred_crop0_process,
        #     os.path.join(gird_path, f"{image_name}_pred_crop0_process.png"),
        #     nrow=batch_size,
        # )

    logging_info(f"Done. Overall {max_iterations} batch, {max_iterations*batch_size} images.")
    recorder.end_recording()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=True, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)