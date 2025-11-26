import torch
from torch import nn
from d2l import torch as d2l

torch.device(type='cuda')
print(torch.cuda.device_count())

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

X = torch.ones(2, 3, device=try_gpu())


