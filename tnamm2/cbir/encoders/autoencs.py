import torch
import torchvision
from torchvision import transforms
from skimage.transform import resize
from .. import utils
from .aemodels import cAutoencoder, DcAutoencoder
import numbers
import numpy as np
from torchvision.transforms.functional import pad

class cOmniAutoEncoder(object):
    def __init__(self):
        device = "cuda" if utils.is_cuda_capable() else "cpu"
        self._device = torch.device(device)
        self.caemodel = cAutoencoder()
        self.caemodel.load_state_dict(torch.load('./model_cae_afa74646.mdl', map_location=device))
        self.caemodel.eval()

    def embedding(self, image):
        image = np.mean(image, axis = 2)
        maxval = image.max()
        minval = image.min()

        image = (image - minval) / (maxval - minval)
        muval = image.mean()
        padding = utils.get_padding(image)
        squimage = np.pad(image, 
            ((padding[0],padding[2]),(padding[1],padding[3])), 
                mode='linear_ramp', end_values = muval)
        #print('shape after padding:', squimage.shape)
        squimage = resize(squimage, (50, 50), anti_aliasing=True)
   
        np.save('tmp.npy', squimage)
        #print('shape after resize:', squimage.shape)
        image = torch.as_tensor(squimage, dtype=torch.float32)
        
        x = image.unsqueeze(0).to(self._device).view(1,1,50,50)
        with torch.no_grad():
            _, code = self.caemodel(x)
            return code


class DcOmniAutoEncoder(object):
    def __init__(self):
        device = "cuda" if utils.is_cuda_capable() else "cpu"
        self._device = torch.device(device)

        self.dcaemodel = DcAutoencoder()
        self.dcaemodel.load_state_dict(torch.load('firstdaemodel.mdl', map_location=device))
  
        self.dcaemodel.eval()

    def embedding(self, image):
        image = np.mean(image, axis = 2)
        maxval = image.max()
        minval = image.min()

        image = (image - minval) / (maxval - minval)
        muval = image.mean()
        padding = utils.get_padding(image)
        squimage = np.pad(image, 
            ((padding[0],padding[2]),(padding[1],padding[3])), 
                mode='linear_ramp', end_values = muval)
        #print('shape after padding:', squimage.shape)
        squimage = resize(squimage, (50, 50), anti_aliasing=True)
   
        #print('shape after resize:', squimage.shape)
        image = torch.as_tensor(squimage, dtype=torch.float32)
        
        x = image.unsqueeze(0).to(self._device).view(1,1,50,50)
        with torch.no_grad():
            _, code = self.dcaemodel(x)
            return code
        
