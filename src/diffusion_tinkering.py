import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import numpy as np
from tqdm import tqdm
from utils import np_mmm_info 

def spawn_models():
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        timesteps = 128
    )

    model.cuda()
    diffusion.cuda()

    return model, diffusion

from utils import img_np_2_pt, img_pt_2_np, load_and_process_img, find_imgs
from utils_visu import imgs_display_plot, plot_imgs_save, display_process
from dataset import training_img_source


def img_tinkering():
    img_path = find_imgs("./dataset")[0]
    img_np = load_and_process_img(img_path)
    img_shape = img_np.shape
    print(f"+++ img_shape: {img_shape}")

    img_pt = img_np.transpose((2, 0, 1)) 
    img_shape = img_pt.shape

    imgs_display_plot([img_np, img_pt_2_np(img_np_2_pt(img_np))])
    imgs_display_plot([img_pt])


def ddpm_tnkering():
    model, ddpm = spawn_models()
    sampled_images = ddpm.sample(batch_size = 1, return_all_timesteps=True)
    result_size = sampled_images.shape # (4, 3, 128, 128)
    print(f"+++ result_size: {result_size}")
    single_process = sampled_images[0]
    display_process(single_process, 16)

def train_tinkering():
    model, diffusion = spawn_models()
    img_list = find_imgs("./dataset")
    training_gen = training_img_source(img_list, batch_size=16, epochs_num=10)
    while True:
        training_batch, progress_info = next(training_gen)
        if training_batch is None:
            print(progress_info)
            break
        
        loss_arr = []
        for _ in tqdm(range(128)):
            loss = diffusion(training_batch)
            loss_arr.append(loss.item())
            loss.backward()

        progress_info = f"+++ {progress_info} | loss: {np_mmm_info(np.array(loss_arr))}"

        sampled_images = diffusion.sample(batch_size=1, return_all_timesteps=True)
        print(progress_info)
        display_process(sampled_images[0], 16, save=True, path="result")

    print("+++ training finished")

# img_tinkering()
# ddpm_tnkering()
# train_tinkering()