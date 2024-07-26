from src.utils import load_and_process_img, img_np_2_pt, img_pt_2_np


import os
import torch

from tqdm import tqdm
from skimage import io



def load_dataset_to_memory(down_scale = 1):
    dataset_dir = "dataset"
    seq_dirs = os.listdir(dataset_dir)
    all_frames = []
    total = 0

    for i, seq_dir in tqdm(enumerate(seq_dirs)):
        dir_seq_full = os.path.join(dataset_dir,seq_dir)
        seq_frames = os.listdir(dir_seq_full)
        
        loaded_frames = []
        for frame_file in seq_frames:
            frame_file_full = os.path.join(dir_seq_full, frame_file)
            loaded_frames.append(load_and_process_img(frame_file_full, scale_down=down_scale))

        # if i > 5:
        #     break
        
        all_frames.append(loaded_frames)
        total += len(seq_frames)

    print(f"+++ total framse: {total}")
    return all_frames

def dataset_np_2_pt(dataset):
    for sequence in dataset:
        frame_num = len(sequence)
        for i in range(frame_num):
            sequence[i] = img_np_2_pt(sequence[i])
        
    return dataset

import math
import random
import numpy as np

def sample_dataset(dataset, batch_size):
    elem_num = len(dataset)
    batch = []

    sample_left = batch_size
    while sample_left > 0:
        sub_sample_num = sample_left
        if sub_sample_num > elem_num:
            sub_sample_num = elem_num
        batch.extend(random.sample(dataset, sub_sample_num))
        sample_left -= sub_sample_num

    for i in range(batch_size):
        batch[i] = random.sample(batch[i], 1)[0]
    
    return batch


def training_img_source(batch_size=32, epochs_num=1, scale_down=16):
    print("nothing?")
    full_dataset = load_dataset_to_memory(down_scale=scale_down)
    full_dataset_pt = dataset_np_2_pt(full_dataset)

    seq_num = len(full_dataset)
    seq_len = len(full_dataset_pt[0])

    data_pnt_num = seq_num*seq_len
    batch_per_epoch = math.ceil(data_pnt_num/batch_size)


    for epoch_idx in range(epochs_num):
        for batch_idx in range(batch_per_epoch):
            batch = sample_dataset(full_dataset_pt, batch_size)

            train_tensor = torch.stack(batch)
            log_info = f"progress: epoch {epoch_idx+1} | batch {batch_idx+1}/{batch_per_epoch}"
            process_info = {"log": log_info, "epoch": epoch_idx}
            batch = []
            yield train_tensor, process_info

    process_info = {"log": "finished", "epoch": epochs_num}
    yield None, process_info