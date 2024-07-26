import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from src.utils import img_pt_2_np

def imgs_display_plot(img_arr):
    img_num = len(img_arr)
    plt.figure(figsize=(10,4)) 
    for i, img in enumerate(img_arr):
        plt.subplot(1, img_num, i + 1)
        plt.imshow(img)
    plt.show()

def plot_imgs_save(img_arr: np.ndarray, path="tmp"):
    single_img = np.concatenate(img_arr, axis=1)
    idx_for_name = len(os.listdir(path))

    idx_for_name = str(idx_for_name).zfill(5)
    io.imsave(f"{path}/train_{idx_for_name}.png", single_img)

def display_process(process_tensor, proces_len, save=False, path="result"):
    choosen_ones = []
    for i, img_pt in enumerate(process_tensor):
        if i % proces_len == proces_len -1:
            img_np = img_pt_2_np(img_pt)
            choosen_ones.append(img_np)

    if save:
        plot_imgs_save(choosen_ones, path)
    else:
        imgs_display_plot(choosen_ones)