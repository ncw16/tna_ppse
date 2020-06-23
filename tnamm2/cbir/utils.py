import time
import datetime
import hashlib
import numpy as np
import torch
import numbers

def get_padding(image): 
    w = image.shape[0]
    h = image.shape[1]
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

def show_progress(func, iterable, **kwargs):
    """
    Wraps a function to privide and expected time of arrival
    func (callable): the function you want to compute
    iterable (iterable): an iterable to loop through

    Use **kwargs to provide additional inputs to the function
    """
    results = []
    times = []
    total = len(iterable)
    for i, item in enumerate(iterable):
        start = time.time()
        results.extend(func(item, **kwargs))
        times.append(time.time() - start)
        avg = np.mean(times)
        eta = avg * total - avg * (i + 1)
        eta = datetime.timedelta(seconds=eta)
        print("Progress %d/%d - ETA: %s" % (i + 1, total, eta), end="\r")
    return results


def get_image_id(array):
    """
    Returns the sha1 hex representation of a numpy array
    see: https://gist.github.com/epignatelli/75cf84b1534a1e817ea36004dfd52e6a
    for performance tests
    """
    return hashlib.sha1(array).hexdigest()


def is_cuda_capable():
    if not torch.cuda.is_available():
        return False

    CUDA_VERSION = torch._C._cuda_getCompiledVersion()
    supported = True
    for d in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(d)
        major = capability[0]
        minor = capability[1]
        supported &= major > 3  # too old
        supported &= CUDA_VERSION <= 9000 and major >= 7 and minor >= 5  # wrong binaries
    return supported
