from src.scheduling import scheduling
from src.utils import find_imgs, img_pt_2_np
from src.dataset import training_img_source, load_dataset_to_memory
from src.utils_visu import imgs_display_plot, plot_imgs_save
from src.unet import SimpleUnet, SinusoidalPositionEmbeddings

import numpy as np
from tqdm import tqdm

import torch
from torch.nn.functional import l1_loss
from torch.optim import Adam

def forward_test_standalone():
    sched = scheduling()
    train_gen = training_img_source(batch_size=1, scale_down=8)
    pt_img, _ = next(train_gen)
    pt_img = pt_img[0]
    plot_forward_save(sched, pt_img)

def backward_test_standalone():
    train_size = (128, 128)

    sched = scheduling()
    model = SimpleUnet()
    model.load("checkpoints/model_00000.pt")
    model.cuda()
    plot_backward_save(sched, model, train_size)

def plot_forward_save(sched: scheduling, pt_img, dir="tmp"):
    steps = 7
    delta = int(sched.timstep_num/steps)
    results : list[np.ndarray] = []
    for i in range(steps):
        t = torch.Tensor([i*delta]).type(torch.int64)
        pt_img, pt_noise = sched.forward_diffusion_sample(pt_img, t)
        results.append(img_pt_2_np(pt_img))
        # results.append(img_pt_2_np(pt_noise))

    # imgs_display_plot(results)
    plot_imgs_save(results, path=dir)


stable_noise = None

@torch.no_grad()
def plot_backward_save(sched: scheduling, model: SimpleUnet, img_size: tuple, dir="tmp"):
    global stable_noise

    sigma = 1
    size = (1, 3, img_size[0], img_size[1])
    if stable_noise is None:
        stable_noise = torch.randn(size)

    img_pt = torch.clone(stable_noise)
    img_pt = img_pt.to("cuda")
    plot_steps_num = 12
    plot_steps = []
    
    steps_idx = [i for i in range(0, sched.timstep_num)]
    steps_idx.reverse()
    plot_points_idx = np.linspace(0, sched.timstep_num, plot_steps_num, dtype=int).tolist()
    for idx in tqdm(steps_idx):
        t = torch.full((1,), idx, dtype=torch.long).to("cuda")
        img_pt = sample_at_timestep(sched, model, img_pt, t)
        if idx in plot_points_idx:
            plot_steps.append(img_pt_2_np(torch.clamp(img_pt, -1.0, 1.0)[0]))

    plot_imgs_save(plot_steps, path=dir)



def sample_at_timestep(sched: scheduling, model: SimpleUnet, x, t):
    betas_t = sched.get_index_from_list(sched.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = sched.get_index_from_list(
        sched.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = sched.get_index_from_list(sched.sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = sched.get_index_from_list(sched.posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 



def get_training_loss(sched: scheduling, model: SimpleUnet, x_0, t):
    # czyli co ostatecznie wiemy, że model próbuje przewidzieć szum całkowity...
    # NO to zaczynają się rodzić juz kolejne pomysły na testy, bo można manipulować tym co on tak na prawdę planuje przewidzieć
    # tak nie naukowo co prawda, w sensie nie zgadza się z wzorem
    x_noisy, noise = sched.forward_diffusion_sample(x_0, t, "cuda")
    noise_pred = model(x_noisy, t)
    return l1_loss(noise, noise_pred)

from matplotlib import pyplot as plt
import os

def save_loss_plot(loss_history: list[float], dir: str = "tmp"):
    filename = f"loss_plot.png"

    avg_samples = []
    samples = len(loss_history)
    avg_over_samples = 10
    for i in range(samples//avg_over_samples):
        start = i*avg_over_samples + 0
        end = i*avg_over_samples + avg_over_samples
        avg_samples.append(sum(loss_history[start:end]))

    plt.clf()
    plt.plot(avg_samples)
    plt.xlabel(f"batch x {avg_over_samples}")
    plt.ylabel("loss")
    plt.savefig(os.path.join(dir,filename))
    plt.clf()


def main():
    sched = scheduling()
    # sched.sched_preview()

    batch_size = 32
    epochs = 100
    trainsteps_per_batch = 100 # multisteps

    dataset_resolution = (1024, 1024)
    train_resolution = (128, 128)
    scale_down = 8

    training_gen = training_img_source(batch_size=batch_size, epochs_num=epochs, scale_down=scale_down)
    pt_img_batch, _ = next(training_gen)

    unet = SimpleUnet().to("cuda")
    # unet.load("checkpoints/model_00032.pt")
    optimizer = Adam(unet.parameters(), lr=0.001)

    loss_history = []
    batch_counter = 0
    batch_per_save = 500
    batch_per_visu = 50
    batch_per_loss = 50


    for train_batch, batch_info in training_gen:
        if train_batch is None:
            print(batch_info["log"])
            break
        
        print("before")
        optimizer.zero_grad()
        t = torch.randint(0, sched.timstep_num, (batch_size,)).long().to("cuda")
        loss = get_training_loss(sched, unet, train_batch, t)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        print("after")

        batch_counter += 1
        if(batch_counter % batch_per_save == 0):
            unet.save("tmp/checkpoints")
        if(batch_counter % batch_per_loss == 0):
            save_loss_plot(loss_history, "tmp/loss")
        if(batch_counter % batch_per_visu == 0):
            plot_backward_save(sched, unet, train_resolution, dir="tmp/result")

        log = batch_info["log"]
        print(f"+++ Loss: {loss_history[-1]} batch: {log}")
        

    unet.save("tmp/checkpoints", name="final")
    print(f"model training finished")


    
# forward_test_standalone()
# backward_test_standalone()



main()

