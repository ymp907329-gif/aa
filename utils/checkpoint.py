import os
import torch


def save_checkpoint(state, save_dir, filename='model.pth'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path, model, optimizer=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0)