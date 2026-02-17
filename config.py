import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CFG:
    TRAIN_NPY: str = ""
    TEST_NPY:  str = ""

    P_IN: int  = 3

    H_ORIG: int = 100
    W_ORIG: int = 100
    C_ORIG: int = 3

    CHANNELS: list = None
    C: int = 3

    USE_CROP: bool = False
    CROP_MODE: str = "center"
    CROP_SIZE: int = 20
    CROP_BOX: Tuple[int, int, int, int] = None

    H: int = 20
    W: int = 20

    BATCH: int = 8
    EPOCHS: int = 50
    LR: float = 5e-4
    WD: float = 1e-3
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True

    D_MODEL: int = 128
    PERIODS: list = None
    NUM_FREQUENCIES: int = 16
    DROPOUT: float = 0.1

    LOSS_DELTA: float = 0.8
    SAVE_DIR: str = "./checkpoints"
    SEED: int = 42

    USE_TGATING: bool = True
    TGATING_GAIN: float = 0.2

    USE_SFE: bool = True

    USE_GOT: bool = True
    N_PROTO: int = 128
    GOT_EPS: float = 0.08
    GOT_TAU: float = 1.4
    GOT_USE_EMB_DIST: bool = True

    USE_MGAT: bool = True
    DLAK_K: int = 32
    MGAT_R0: float = 1.5

    USE_QLOSS: bool = True
    Q_W: float = 0.3

    USE_RESIDUAL_TO_LAST: bool = True

    NODE_MODE: bool = True
    COORDS_NPY: str = None

cfg = CFG()

if cfg.PERIODS is None:
    cfg.PERIODS = [48, 168]

