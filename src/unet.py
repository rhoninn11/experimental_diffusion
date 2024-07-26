import torch
from torch import nn
import math

import os

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        # print(embeddings)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # print(embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        # print(embeddings)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # print(embeddings)
        # TODO: Double check the ordering here
        return embeddings

class TimeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time_emb_dim = 32

        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
                nn.ReLU()
            )
        
    def forward(self, timestep):
        return self.time_mlp(timestep)
    

def size_options(opt):
    down = (64, 128, 256, 512, 1024)
    up = (1024, 512, 256, 128, 64)

    if opt == 1:
        down = (64, 96, 128, 192, 256, 384, 512)
        up = (512, 384, 256, 192, 128, 96, 64)

    return down, up




class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        # model_01 model_2
        down_chan_list, up_chan_list = size_options(0),

        depth = len(down_chan_list)
        assert depth == len(up_chan_list)

        out_chan = 3 
        self.time_mlp = TimeMLP()
        time_emb_dim = self.time_mlp.time_emb_dim
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_chan_list[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_chan_list[i], down_chan_list[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_chan_list)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_chan_list[i], up_chan_list[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_chan_list)-1)])
        
        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_chan_list[-1], out_chan, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)
    

    def save(self, path, name = None):
        file_pos = len(os.listdir(path))
        filename = f"model_{file_pos:04d}.pt"
        if name:
            filename = f"{name}.pt"
        file = os.path.join(path, file)
        torch.save(self.state_dict(), filename)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
