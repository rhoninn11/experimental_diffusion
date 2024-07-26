import os
import torch
from skimage import io
import numpy as np

def np_mmm_info(np_array):
    max_val = np_array.max()
    min_val = np_array.min()
    mean_val = np_array.mean()
    return f"mean - {mean_val}, min - {min_val}, max - {max_val}"


def img_np_2_pt(img_np, one_minus_one=True, transpose=True):
    if transpose:
        img_np = img_np.transpose((2, 0, 1))

    img_np = (img_np / 255.0)
    if one_minus_one:
        img_np = img_np * 2 - 1
    img_pt = torch.from_numpy(img_np).float().cuda()
    
    return img_pt

def img_pt_2_np(img_pt, one_minus_one=True, transpose=True) -> np.ndarray:
    img_np = img_pt.cpu().detach().numpy()
    # to uint 8
    if one_minus_one:
        img_np = img_np.clip(-1, 1)*0.5 + 0.5
    else:
        img_np = img_np.clip(0, 1)

    img_np = img_np * 255.0
    img_np = img_np.astype("uint8")
    if transpose:
        img_np = img_np.transpose((1, 2, 0))

    return img_np

def img_downscale(img_np, downscale):
    img_np = img_np[::downscale, ::downscale, :]
    return img_np

def reduce_channels(img_np):
    img_np = img_np[:, :, :3]
    return img_np

def load_and_process_img(img_path, scale_down=16):
    img_np = io.imread(img_path)
    img_np = img_downscale(img_np, scale_down)
    img_np = reduce_channels(img_np)
    return img_np


def find_imgs(path):
    file_list = []
    files_in = os.listdir(path)
    for file in files_in:
        file_list.append(os.path.join(path, file))

    # print(f"+++ files smaple {file_list[:5]}")
    return file_list