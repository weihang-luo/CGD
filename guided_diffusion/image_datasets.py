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

import random
import os

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import re
import torch

# 训练用
def load_data(
        data_dir,
        batch_size,
        class_dict,
        shuffle=False, # 是否打乱
        crop_size=None,
        crop_margins=None,
):
    paths = data_dir.glob('**/*.jpg')

    images_path = []
    # labels = []
    for img in paths:
        images_path.append(img)

    if crop_size is not None:
        dataset = RandomCroppedImageDataset(images_path, class_dict, crop_size, crop_margins)
    else:
        dataset = ImageDataset(images_path, class_dict)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True
    )
    while True:
        yield from loader

class ImageDataset(Dataset):
    def __init__(self, image_path, classes, transform=None):
        super().__init__()
        self.image_path = image_path
        self.classes = classes
        self.transform = transform
    
    def __getitem__(self, index):
        img_path = self.image_path[index]
        img = Image.open(img_path)
        img = img.convert('RGB')

        arr = np.asarray(img)
        arr = arr.astype(np.float32) / 127.5 - 1
        ts = torch.from_numpy(arr).permute(2, 0, 1)
        y = self.classes[img_path.stem.split('_')[0]]
        return ts, y
    
    def __len__(self):
        return len(self.image_path)

class RandomCroppedImageDataset(Dataset):
    def __init__(self, image_path, classes, crop_size, crop_margins, transform=None):
        super().__init__()
        self.image_path = image_path
        self.classes = classes
        self.crop_size = crop_size  # Set the crop size during initialization
        self.transform = transform
        self.crop_margins = crop_margins
    
    def __getitem__(self, index):
        img_path = self.image_path[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        label = img_path.stem.split('_')[0]
        y = self.classes[label]
        # Convert the image to a NumPy array
        arr = np.asarray(img).astype(np.float32) / 127.5 - 1

        # Get image dimensions
        h, w, _ = arr.shape

        # Exclude 20 pixels from each edge
        crop_margin = self.crop_margins[label]
        max_h = h - self.crop_size - crop_margin
        max_w = w - self.crop_size - crop_margin

        # Ensure the crop size is valid
        if max_h <= crop_margin or max_w <= crop_margin:
            raise ValueError(f"Crop size {self.crop_size} is too large for the image dimensions.")

        # Randomly select top-left corner within allowed region
        top = random.randint(crop_margin, max_h)
        left = random.randint(crop_margin, max_w)

        # Crop the image
        cropped_arr = arr[top:top + self.crop_size, left:left + self.crop_size, :]

        # Convert to tensor and permute dimensions to [C, H, W]
        ts = torch.from_numpy(cropped_arr).permute(2, 0, 1)
        y = 0
        return ts, y
    
    def __len__(self):
        return len(self.image_path)



class JointImageDataset(Dataset):
    def __init__(self, image_path, classes, crop_size=None, crop_class_dict=None, transform=None):
        super().__init__()
        self.image_path = image_path
        self.classes = classes
        self.crop_size = crop_size
        self.crop_class_dict = crop_class_dict
        self.transform = transform

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        img_path = self.image_path[index]

        defect_class = img_path.stem.split("_")[0]
        pos_l = img_path.stem.split("_")[2].split("-")
        pos = np.array([int(p) for p in pos_l])
        img_class = img_path.stem.split("_")[3][1:] 

        img = Image.open(img_path)
        img = img.convert('RGB')
        arr = np.asarray(img)
        arr = arr.astype(np.float32) / 127.5 - 1
        crop = arr[pos[1]:pos[1]+self.crop_size, pos[0]:pos[0]+self.crop_size, :].copy()  # numpy使用copy实现独立拷贝

        label = self.classes[img_class]
        crop_label = self.crop_class_dict[defect_class]

        return {
            'x': np.transpose(arr, [2, 0, 1]),
            'label': label,
            'crop': np.transpose(crop, [2, 0, 1]),
            'crop_label': crop_label,
            'pos': pos,   # 位置
        }
    

# 推理用
def load_data_yield(loader):
    while True:
        yield from loader

def load_data_inpa(
    *,
    gt_path=None,
    batch_size,
    image_size,
    deterministic=False,
    return_dataloader=False,
    return_dict=False,
    max_len=None,
    drop_last=True,
    classes=None,
    offset=0,
    location=None,
    ** kwargs
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    gt_dir = os.path.expanduser(gt_path)
    gt_paths = _list_image_files_recursively(gt_dir)

    dataset = ImageDatasetInpa(
        image_size,
        gt_paths=gt_paths,
        shard=0,
        num_shards=1,
        return_dict=return_dict,
        max_len=max_len,
        offset=offset,
        classes=None,
        location=location,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False if deterministic else True, num_workers=1, drop_last=drop_last
    )
    if return_dataloader:
        return loader
    else:
        return load_data_yield(loader)

class ImageDatasetInpa(Dataset):
    def __init__(
        self,
        resolution,
        gt_paths,
        shard=0,
        num_shards=1,
        return_dict=False,
        max_len=None,
        offset=0,
        classes=None,
        location=None,
    ):
        super().__init__()
        self.resolution = resolution
        gt_paths = sorted(gt_paths)[offset:]
        # gt_paths[shard:]表示从索引shard开始切片，返回一个新的子列表。
        # [::num_shards]对上一步得到的子列表再次进行切片，使用步长num_shards选择元素。
        self.local_gts = gt_paths[shard:][::num_shards]
        self.return_dict = return_dict
        self.max_len = max_len
        self.location = location
        self.classes=classes,
            
    def __len__(self):
        if self.max_len is not None:
            return self.max_len

        return len(self.local_gts)

    def __getitem__(self, idx):

        gt_path = self.local_gts[idx]
        pil_gt = self.imread(gt_path)

        arr_gt = np.asarray(pil_gt)
        gt_l = np.asarray(pil_gt).astype(np.float32) / 127.5 - 1
        
        exp_arr_gt = np.zeros((self.resolution, self.resolution, arr_gt.shape[2]), dtype=np.uint8)
        exp_arr_mask = np.zeros((self.resolution, self.resolution, arr_gt.shape[2]), dtype=np.uint8)

        # 选择随机位置
        if self.location:
            y = int(os.path.basename(gt_path).split('_')[1])
            lx, ly =self.location
            lx = lx[y]
        elif re.search(r'x(\d+)y(\d+)', gt_path):
            match = re.search(r'x(\d+)y(\d+)', gt_path)
            lx = int(match.group(1))  # 获取 x 后的数值
            ly = int(match.group(2))  # 获取 y 后的数值
        else:
            lx = np.random.randint(0, self.resolution - arr_gt.shape[1] + 1)
            ly = np.random.randint(0, self.resolution - arr_gt.shape[0] + 1)
        
        exp_arr_gt[ly : ly+arr_gt.shape[0], lx : lx+arr_gt.shape[1], :] = arr_gt
        exp_arr_gt = exp_arr_gt.astype(np.float32) / 127.5 - 1
        exp_arr_mask[ly : ly+arr_gt.shape[0], lx : lx+arr_gt.shape[1], :] = 225.0
        exp_arr_mask = exp_arr_mask.astype(np.float32) / 255.0
        name = os.path.splitext(os.path.basename(gt_path))[0]

        return {
            'GT': np.transpose(exp_arr_gt, [2, 0, 1]),
            'mask': np.transpose(exp_arr_mask, [2, 0, 1]),   # 黑白蒙版
            'local': np.transpose(gt_l, [2, 0, 1]),
            'GT_name': name,
            'GT_size': arr_gt.shape[1],  
        }

    def imread(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# 缺陷对应的电路板类型分类训练使用
def load_data_sample_class(
        data_dir,
        batch_size,
        class_dict,
        shuffle=False, # 是否打乱
):
    paths = data_dir.glob('**/*.jpg')

    images_path = []
    # labels = []
    for img in paths:
        images_path.append(img)
    #     labels.append(img.stem.split('_')[0])

    # classes = {label:int(i) for i,label in enumerate(sorted(set(labels)))}

    # with classes_json.open('w') as f:
    #     json.dump(classes, f)

    dataset = ImageDatasetSampleClass(images_path, class_dict)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True
    )
    while True:
        yield from loader

class ImageDatasetSampleClass(Dataset):
    def __init__(self, image_path, classes, transform=None):
        super().__init__()
        self.image_path = image_path
        self.classes = classes
        self.transform = transform
    
    def __getitem__(self, index):
        img_path = self.image_path[index]
        img = Image.open(img_path)
        img = img.convert('RGB')

        arr = np.asarray(img)
        arr = arr.astype(np.float32) / 127.5 - 1
        ts = torch.from_numpy(arr).permute(2, 0, 1)
        y = self.classes[img_path.stem.split('_')[1][1:]]
        return ts, y
    
    def __len__(self):
        return len(self.image_path)
