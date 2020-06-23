import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable

# Believe we need this for the bit in between coder and decoder
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 40, 6, 6)

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.double).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
#            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            #sampled_noise = self.noise.repeat(*x.size()).normal_() * self.sigma
            sampled_noise = self.sigma*torch.randn_like(x)
            x = x + sampled_noise
        return x 

class DcAutoencoder(nn.Module):
    def __init__(self, code_size=64):
        super(DcAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            GaussianNoise(),
            nn.Conv2d(1, 20, kernel_size=5, stride=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(True),
            nn.Conv2d(20, 40, kernel_size=3, stride=2, bias=False),
           # nn.BatchNorm2d(40),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.LeakyReLU(True),
            nn.Flatten(),
            nn.Linear(1440, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 1440, bias=True),
            nn.ReLU(True),
            UnFlatten(),
            nn.ConvTranspose2d(40, 40, kernel_size=3, stride=1, dilation=2, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(True),
            nn.ConvTranspose2d(40, 20, kernel_size=6, stride=2, dilation=1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.ConvTranspose2d(20, 1, kernel_size=4, stride=2, dilation=1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        code = self.encoder(x)
        foldedcode = code 
        xdash = self.decoder(foldedcode)

        return xdash, code


class cAutoencoder(nn.Module):
    def __init__(self, code_size=64):
        super(cAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, bias=True),
            nn.LeakyReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(True),
            nn.Conv2d(20, 40, kernel_size=3, stride=2, bias=False),
           # nn.BatchNorm2d(40),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.LeakyReLU(True),
            nn.Flatten(),
            nn.Linear(1440, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 1440, bias=True),
            nn.ReLU(True),
            UnFlatten(),
            nn.ConvTranspose2d(40, 40, kernel_size=3, stride=1, dilation=2, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(True),
            nn.ConvTranspose2d(40, 20, kernel_size=6, stride=2, dilation=1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.ConvTranspose2d(20, 1, kernel_size=4, stride=2, dilation=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        code = self.encoder(x)
        foldedcode = code 
        xdash = self.decoder(foldedcode)

        return xdash, code

