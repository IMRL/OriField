import numpy as np
import torch


def peroidic_np(a, norm=True, abs=False):
    if norm:
        a = a / np.pi
    mod = a % 2
    peroid = np.where(mod > 1, mod - 2, mod)
    if abs:
        peroid = np.abs(peroid)
    return peroid


def peroidic_torch(a, norm=True, abs=False):
    if norm:
        a = a / np.pi
    mod = a % 2
    peroid = torch.where(mod > 1, mod - 2, mod)
    if abs:
        peroid = torch.abs(peroid)
    return peroid
