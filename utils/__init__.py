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

import yaml
import os
from PIL import Image
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

from .logger import logging_info

def txtread(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        return f.read()


def yamlread(path):
    return yaml.safe_load(txtread(path=path))

def imwrite(path=None, img=None):
    Image.fromarray(img).save(path)


def normalize_image(tensor_img):
    tensor_img = (tensor_img + 1.0) / 2.0
    return tensor_img


def save_grid(tensor_img, path, nrow=4):
    """
    tensor_img: [B, 3, H, W] or [tensor(3, H, W)]
    """
    if isinstance(tensor_img, list):
        tensor_img = torch.stack(tensor_img)
    assert len(tensor_img.shape) == 4
    tensor_img = tensor_img.clamp(min=0.0, max=1.0)
    grid = make_grid(tensor_img, nrow=nrow)
    pil = ToPILImage()(grid)
    pil.save(path)


def save_image(tensor_img, path):
    """
    tensor_img : [3, H, W]
    """
    tensor_img = tensor_img.clamp(min=0.0, max=1.0)
    pil = ToPILImage()(tensor_img)
    pil.save(path)
