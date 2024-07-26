
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as ptf
import numpy as np

class scheduling():
    def __init__(self):
        self.timstep_num = 255
        self.betas = self.linear_beta_schedule(timesteps=self.timstep_num)

        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.concat((torch.asarray([1.0]), self.alphas_cumprod[:-1]))
        print(f"+++ alphas_cumprod_prev: {self.alphas_cumprod_prev.shape}")
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    def sched_preview(self):
        all_precomputed = [
            self.betas, self.alphas, self.alphas_cumprod, 
            self.alphas_cumprod_prev, self.sqrt_recip_alphas, 
            self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, 
            self.posterior_variance]
        
        plot_num = len(all_precomputed)
        x = torch.linspace(1,self.timstep_num, self.timstep_num)

        plt.figure(figsize=(10,4)) 
        for i, plot in enumerate(all_precomputed):
            plt.subplot(1, plot_num, i + 1)
            plt.plot(x, plot)
        plt.show()



    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        ramp = torch.linspace(0.5,1, timesteps)
        ramp = ramp*ramp + 0.3
        return ramp*torch.linspace(start, end, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0: torch.Tensor, t: torch.Tensor, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        sigma = 1
        noise = torch.randn(x_0.shape, device=device)
        noise_offset = torch.randn((1,), device=device)

        # histogram = torch.histogram(noise.cpu(), bins=40, range=(-9, 9))
        # hist = histogram.hist.tolist()
        # bins = histogram.bin_edges.tolist()
        # bins = (np.array(bins[:-1]) + np.array(bins[1:]))/2
        # bins = bins.tolist()
        # print(f"len of hist: {len(hist)} and bins: {len(bins)}")
        # plt.plot(hist, bins)
        # plt.show()
        noise = noise * 0.9 + noise_offset * 0.1

        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


