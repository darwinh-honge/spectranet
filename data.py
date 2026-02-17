import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

from config import cfg

# -------------------------
# seed / device
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(x):
    return x.to(cfg.DEVICE, non_blocking=True)

# -------------------------
# shape utils
# -------------------------
def standardize_to_grid_or_node4d(a: np.ndarray) -> np.ndarray:
    if a.ndim == 4:
        return a
    if a.ndim == 3:
        T, N, C = a.shape
        return a.reshape(T, 1, N, C)
    if a.ndim == 2:
        T, N = a.shape
        return a.reshape(T, 1, N, 1)
    raise ValueError(f"Unsupported input shape: {a.shape}")

def is_grid4d(a: np.ndarray) -> bool:
    return (a.ndim == 4 and a.shape[1] > 1 and a.shape[2] > 1)

# -------------------------
# crop / channel select
# -------------------------
def crop_data(arr: np.ndarray, crop_mode: str, crop_size: int, seed: int = None, crop_box: Tuple[int,int,int,int] = None) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D, got {arr.shape}")
    T, H_orig, W_orig, C = arr.shape

    if crop_box is not None:
        h0, h1, w0, w1 = map(int, crop_box)
        h0 = int(max(0, min(h0, H_orig)))
        h1 = int(max(0, min(h1, H_orig)))
        w0 = int(max(0, min(w0, W_orig)))
        w1 = int(max(0, min(w1, W_orig)))
        return arr[:, h0:h1, w0:w1, :]

    if crop_mode == "none":
        return arr
    if crop_size > min(H_orig, W_orig):
        return arr

    if crop_mode == "center":
        center_h, center_w = H_orig // 2, W_orig // 2
        half_crop = crop_size // 2
        start_h = center_h - half_crop
        start_w = center_w - half_crop
    elif crop_mode == "top_left":
        start_h, start_w = 0, 0
    elif crop_mode == "bottom_right":
        start_h = H_orig - crop_size
        start_w = W_orig - crop_size
    elif crop_mode == "random":
        if seed is not None:
            np.random.seed(seed)
        start_h = np.random.randint(0, H_orig - crop_size + 1)
        start_w = np.random.randint(0, W_orig - crop_size + 1)
    else:
        raise ValueError(f"Unknown crop_mode: {crop_mode}")

    start_h = max(0, min(start_h, H_orig - crop_size))
    start_w = max(0, min(start_w, W_orig - crop_size))
    end_h = start_h + crop_size
    end_w = start_w + crop_size
    return arr[:, start_h:end_h, start_w:end_w, :]

def select_channels(arr: np.ndarray, channels: list, channel_names: list = None) -> np.ndarray:
    return arr[..., channels]

# -------------------------
# windowing
# -------------------------
def sliding_windows(arr: np.ndarray, p_in: int) -> Tuple[np.ndarray, np.ndarray]:
    T = arr.shape[0]
    N = T - p_in
    X = np.stack([arr[i:i+p_in] for i in range(N)], axis=0)
    y = np.stack([arr[i+p_in] for i in range(N)], axis=0)
    return X, y

# -------------------------
# normalization
# -------------------------
def norm_fit(x_train: np.ndarray):
    min_v = x_train.min(axis=(0,1,2), keepdims=True)
    max_v = x_train.max(axis=(0,1,2), keepdims=True)
    return min_v, max_v

def norm_apply(x: torch.Tensor, min_v: torch.Tensor, max_v: torch.Tensor, eps: float = 1e-6):
    scale = (max_v - min_v)
    scale = torch.where(scale < eps, torch.ones_like(scale), scale)
    return (x - min_v) / (scale + eps)

def denorm(x: torch.Tensor, min_v: torch.Tensor, max_v: torch.Tensor, eps: float = 1e-6):
    return min_v + x * (max_v - min_v + eps)

# -------------------------
# coords + knn
# -------------------------
def make_grid_coords_np(H: int, W: int) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, H, dtype=np.float32),
        np.linspace(-1, 1, W, dtype=np.float32),
        indexing="ij"
    )
    coords = np.stack([yy, xx], axis=-1).reshape(-1, 2)
    return coords.astype(np.float32)

def make_line_coords_np(N: int) -> np.ndarray:
    x = np.linspace(-1, 1, N, dtype=np.float32)
    y = np.zeros_like(x)
    return np.stack([y, x], axis=-1).astype(np.float32)

def build_knn_from_coords(coords: torch.Tensor, k: int):
    N = coords.size(0)
    d2 = torch.cdist(coords, coords, p=2.0)**2
    d2 = d2 + torch.eye(N, device=coords.device) * 1e9
    idx = torch.topk(-d2, k, dim=-1).indices
    rel = coords.unsqueeze(1) - coords[idx]
    geo = rel.norm(dim=-1)
    return idx, rel, geo

def build_knn_idx_and_relpos(H, W, k, device):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    coords = torch.stack([yy, xx], dim=-1).reshape(-1,2).float()
    N = coords.size(0)
    d2 = torch.cdist(coords, coords, p=2.0)**2
    d2 = d2 + torch.eye(N, device=device)*1e9
    idx = torch.topk(-d2, k, dim=-1).indices
    rel = coords.unsqueeze(1) - coords[idx]
    rel[...,0] = 2*rel[...,0]/max(1,(H-1))
    rel[...,1] = 2*rel[...,1]/max(1,(W-1))
    geo = rel.norm(dim=-1)
    return idx, rel, geo, coords

# -------------------------
# Dataset
# -------------------------
class WinDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

