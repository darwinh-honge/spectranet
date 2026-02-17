import os, json, time
import numpy as np
import torch
import torch.nn as nn

from config import cfg
from data import (
    set_seed, to_device,
    standardize_to_grid_or_node4d, is_grid4d,
    crop_data, select_channels, sliding_windows,
    norm_fit, norm_apply, denorm,
    make_grid_coords_np, make_line_coords_np,
    WinDataset
)
from model import SpecTraNet, compute_gating_signal

import warnings
warnings.filterwarnings("ignore")

# -------------------------
# metrics / losses
# -------------------------
def compute_metrics(y_true, y_pred, eps=1e-8):
    B,H,W,C = y_true.shape
    yt = y_true.reshape(B,-1,C); yp = y_pred.reshape(B,-1,C)
    diff = yp - yt
    mse = (diff**2).mean(dim=1).mean(dim=0)
    rmse = torch.sqrt(mse + eps)
    mae  = diff.abs().mean(dim=1).mean(dim=0)
    smape= ( (yp-yt).abs() / (yt.abs()+yp.abs()+eps) ).mean(dim=1).mean(dim=0)
    yt_mean = yt.mean(dim=1, keepdim=True)
    ss_tot  = ((yt-yt_mean)**2).sum(dim=1).mean(dim=0)
    ss_res  = (diff**2).sum(dim=1).mean(dim=0)
    r2 = 1 - ss_res/(ss_tot+eps)
    rmse_g = torch.sqrt(((diff**2).mean()))
    mae_g  = diff.abs().mean()
    smape_g= ( (yp-yt).abs() / (yt.abs()+yp.abs()+eps) ).mean()
    r2_g   = 1 - (diff**2).sum() / (((yt-yt.mean())**2).sum()+eps)
    return {
        "per_channel": {
            "rmse": rmse.detach().cpu().tolist(),
            "mae":  mae.detach().cpu().tolist(),
            "smape": smape.detach().cpu().tolist(),
            "r2":   r2.detach().cpu().tolist(),
        },
        "global": {
            "rmse": float(rmse_g.detach().cpu()),
            "mae":  float(mae_g.detach().cpu()),
            "smape": float(smape_g.detach().cpu()),
            "r2":   float(r2_g.detach().cpu()),
        }
    }

def freq_weighted_loss(y, mu, eps=1e-8):
    yf = torch.fft.rfft2(y.permute(0, 3, 1, 2))
    mf = torch.fft.rfft2(mu.permute(0, 3, 1, 2))
    df = yf - mf
    H = y.shape[1]; W = y.shape[2]
    Hf, Wf = df.shape[2], df.shape[3]
    yy = torch.linspace(-1, 1, H, device=y.device).unsqueeze(1).expand(H, W)
    xx = torch.linspace(-1, 1, W, device=y.device).unsqueeze(0).expand(H, W)
    r = torch.sqrt(yy**2 + xx**2)[:Hf, :Wf]
    w_low = torch.exp(-3.0 * r)
    Wmap = w_low.unsqueeze(0).unsqueeze(0)
    fw = (Wmap * df.abs()).mean()
    return fw

def pinball_loss(y, yhat, tau):
    return torch.maximum(tau*(y - yhat), (tau-1)*(y - yhat)).mean()

# -------------------------
# train / eval
# -------------------------
def train_one_epoch(model, loader, min_v, max_v, optim):
    model.train()
    loss_fn = nn.HuberLoss(delta=cfg.LOSS_DELTA)
    total = 0.0
    for X,y in loader:
        X = to_device(X); y = to_device(y)
        gating_hint = compute_gating_signal(X) if cfg.USE_TGATING else None
        Xn = norm_apply(X, min_v, max_v)
        yn = norm_apply(y, min_v, max_v)
        yp, reg_loss = model(Xn, gating_hint=gating_hint)

        huber = loss_fn(yp, yn)
        fw = freq_weighted_loss(yn, yp) if cfg.USE_SFE else torch.tensor(0.0, device=yp.device)
        q_loss = (pinball_loss(yn, yp, 0.2) + pinball_loss(yn, yp, 0.5) + pinball_loss(yn, yp, 0.8)) / 3.0 if cfg.USE_QLOSS else torch.tensor(0.0, device=yp.device)
        loss = huber + reg_loss + 0.1 * fw + cfg.Q_W * q_loss

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        total += float(loss.detach().cpu())
    return total/len(loader)

@torch.no_grad()
def eval_model(model, loader, min_v, max_v):
    model.eval()
    loss_fn = nn.HuberLoss(delta=cfg.LOSS_DELTA)
    total = 0.0
    all_pred, all_true = [], []
    for X, y in loader:
        X = to_device(X); y = to_device(y)
        gating_hint = compute_gating_signal(X) if cfg.USE_TGATING else None
        Xn = norm_apply(X, min_v, max_v)
        yn = norm_apply(y, min_v, max_v)
        yp, reg_loss = model(Xn, gating_hint=gating_hint)

        huber = loss_fn(yp, yn)
        fw = freq_weighted_loss(yn, yp) if cfg.USE_SFE else torch.tensor(0.0, device=yp.device)
        q_loss = (pinball_loss(yn, yp, 0.2) + pinball_loss(yn, yp, 0.5) + pinball_loss(yn, yp, 0.8)) / 3.0 if cfg.USE_QLOSS else torch.tensor(0.0, device=yp.device)
        loss = huber + reg_loss + 0.1 * fw + cfg.Q_W * q_loss

        total += float(loss.detach().cpu())
        ypd = denorm(yp, min_v, max_v)
        all_pred.append(ypd.cpu())
        all_true.append(y.cpu())

    y_pred = torch.cat(all_pred, dim=0)
    y_true = torch.cat(all_true, dim=0)
    metrics = compute_metrics(y_true, y_pred)
    return total / len(loader), metrics

# -------------------------
# main
# -------------------------
def main():
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    set_seed(cfg.SEED)

    print("Device:", cfg.DEVICE)
    print("SpecTra-Net (Grid+Node unified, SFE, TGATING, GOT, MGAT)")
    print("Loading data...")

    tr_raw = np.load(cfg.TRAIN_NPY)
    te_raw = np.load(cfg.TEST_NPY)
    print(f"Raw train shape: {tr_raw.shape}")
    print(f"Raw test  shape: {te_raw.shape}")

    raw_is_grid = is_grid4d(tr_raw)
    tr_full = standardize_to_grid_or_node4d(tr_raw)
    te_full = standardize_to_grid_or_node4d(te_raw)

    print(f"Standardized train shape: {tr_full.shape}")
    print(f"Standardized test  shape: {te_full.shape}")

    cfg.H_ORIG = tr_full.shape[1]
    cfg.W_ORIG = tr_full.shape[2]
    cfg.C_ORIG = tr_full.shape[3]

    if cfg.C_ORIG == 3:
        channel_names_all = ["SMS", "Call", "Internet"]
    else:
        channel_names_all = [f"Ch{i}" for i in range(cfg.C_ORIG)]

    if cfg.CHANNELS is None:
        cfg.CHANNELS = list(range(cfg.C_ORIG))
    cfg.C = len(cfg.CHANNELS)
    selected_channel_names = [channel_names_all[i] for i in cfg.CHANNELS]
    print(f"Channels: {cfg.CHANNELS} ({selected_channel_names})")

    # chronological split: 9:1 from training period
    T_total = tr_full.shape[0]
    T_tr = int(T_total * 0.9)
    tr_train = tr_full[:T_tr]
    tr_val   = tr_full[T_tr:]

    tr_train = select_channels(tr_train, cfg.CHANNELS, channel_names_all)
    tr_val   = select_channels(tr_val,   cfg.CHANNELS, channel_names_all)
    te       = select_channels(te_full,  cfg.CHANNELS, channel_names_all)

    # crop only when grid
    if (not cfg.USE_CROP) or (not is_grid4d(tr_train)):
        crop_str = "full" if not cfg.USE_CROP else "skipCrop(node)"
        tr_train_p = tr_train
        tr_val_p   = tr_val
        te_p       = te
        if cfg.USE_CROP and (not is_grid4d(tr_train)):
            print("NOTE: Input is node-view; cropping is skipped safely.")
    else:
        if cfg.CROP_BOX is None:
            crop_str = f"{cfg.CROP_MODE}{cfg.CROP_SIZE}"
            tr_train_p = crop_data(tr_train, cfg.CROP_MODE, cfg.CROP_SIZE, seed=cfg.SEED, crop_box=None)
            tr_val_p   = crop_data(tr_val,   cfg.CROP_MODE, cfg.CROP_SIZE, seed=cfg.SEED, crop_box=None)
            te_p       = crop_data(te,       cfg.CROP_MODE, cfg.CROP_SIZE, seed=cfg.SEED, crop_box=None)
        else:
            h0,h1,w0,w1 = cfg.CROP_BOX
            crop_str = f"boxH{h0}-{h1}_W{w0}-{w1}"
            tr_train_p = crop_data(tr_train, "box", cfg.CROP_SIZE, seed=cfg.SEED, crop_box=cfg.CROP_BOX)
            tr_val_p   = crop_data(tr_val,   "box", cfg.CROP_SIZE, seed=cfg.SEED, crop_box=cfg.CROP_BOX)
            te_p       = crop_data(te,       "box", cfg.CROP_SIZE, seed=cfg.SEED, crop_box=cfg.CROP_BOX)

    coords_t = None

    # node-mode handling
    if cfg.NODE_MODE:
        if tr_train_p.shape[1] == 1:
            N = int(tr_train_p.shape[2])
            print(f"NODE_MODE=ON: already node-view (T,1,{N},C)")
        else:
            Hc, Wc = int(tr_train_p.shape[1]), int(tr_train_p.shape[2])
            N = Hc * Wc
            print(f"NODE_MODE=ON: grid->node (T,{Hc},{Wc},C) -> (T,1,{N},C)")

            def grid_to_node_view(a):
                T = a.shape[0]
                return a.reshape(T, 1, Hc*Wc, a.shape[-1])

            tr_train_p = grid_to_node_view(tr_train_p)
            tr_val_p   = grid_to_node_view(tr_val_p)
            te_p       = grid_to_node_view(te_p)

        if cfg.COORDS_NPY is not None:
            coords_np = np.load(cfg.COORDS_NPY).astype(np.float32)
            if coords_np.ndim != 2 or coords_np.shape[0] != N:
                raise ValueError(f"COORDS_NPY must be (N,2) with N={N}, got {coords_np.shape}")
            if coords_np.shape[1] != 2:
                coords_np = coords_np[:, :2]
        else:
            if raw_is_grid and (tr_raw.ndim == 4 and tr_raw.shape[1] > 1 and tr_raw.shape[2] > 1):
                coords_np = make_grid_coords_np(Hc, Wc) if ('Hc' in locals() and 'Wc' in locals()) else make_line_coords_np(N)
            else:
                coords_np = make_line_coords_np(N)

        coords_t = torch.from_numpy(coords_np)
        cfg.H = 1
        cfg.W = N
    else:
        if tr_train_p.shape[1] == 1:
            raise ValueError("NODE_MODE=False but input is node-view (T,1,N,C). Set NODE_MODE=True for node data.")
        cfg.H = int(tr_train_p.shape[1])
        cfg.W = int(tr_train_p.shape[2])
        print(f"NODE_MODE=OFF: using grid view (T,{cfg.H},{cfg.W},C)")

    # windowing
    Xtr,  ytr  = sliding_windows(tr_train_p, cfg.P_IN)
    Xval, yval = sliding_windows(tr_val_p,   cfg.P_IN)
    Xte,  yte  = sliding_windows(te_p,       cfg.P_IN)

    # normalization
    min_np, max_np = norm_fit(tr_train_p)
    min_v = torch.from_numpy(min_np.astype(np.float32)).to(cfg.DEVICE)
    max_v = torch.from_numpy(max_np.astype(np.float32)).to(cfg.DEVICE)

    # dataloaders
    from torch.utils.data import DataLoader
    ds_tr  = WinDataset(Xtr,  ytr)
    ds_val = WinDataset(Xval, yval)
    ds_te  = WinDataset(Xte,  yte)

    dl_tr  = DataLoader(ds_tr,  batch_size=cfg.BATCH, shuffle=True,  num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    dl_val = DataLoader(ds_val, batch_size=cfg.BATCH, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    dl_te  = DataLoader(ds_te,  batch_size=cfg.BATCH, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)

    print(f"Creating SpecTra-Net: H={cfg.H}, W={cfg.W}, C={cfg.C}, P={cfg.P_IN} | crop={crop_str} | NODE_MODE={cfg.NODE_MODE}")

    model = SpecTraNet(cfg.H, cfg.W, cfg.C, cfg.P_IN, coords=coords_t).to(cfg.DEVICE)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.LR, weight_decay=cfg.WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.EPOCHS)

    # naming
    if len(cfg.CHANNELS) == 1:
        channel_str = f"ch{cfg.CHANNELS[0]}"
    elif len(cfg.CHANNELS) == cfg.C_ORIG:
        channel_str = "allch"
    else:
        channel_str = f"ch{'_'.join(map(str, cfg.CHANNELS))}"

    model_name = f"spectranet_{crop_str}_{channel_str}"
    model_name += "_NODE.pt" if cfg.NODE_MODE else ".pt"
    best_path = os.path.join(cfg.SAVE_DIR, model_name)

    best_rmse = 1e9
    print("\n=== Starting Training ===")
    print(f"Model will be saved as: {model_name}")

    for ep in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, dl_tr, min_v, max_v, optim)
        va_loss, va_metrics = eval_model(model, dl_val, min_v, max_v)
        scheduler.step()

        rmse_g = va_metrics["global"]["rmse"]
        r2_g   = va_metrics["global"]["r2"]

        if rmse_g < best_rmse:
            best_rmse = rmse_g
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "min": min_v.cpu().numpy(),
                "max": max_v.cpu().numpy(),
                "channels": cfg.CHANNELS,
                "channel_names": selected_channel_names,
                "coords_npy": cfg.COORDS_NPY
            }, best_path)

        print(f"[Ep {ep:03d}] train={tr_loss:.4f} | val={va_loss:.4f} | "
              f"RMSEg={rmse_g:.4f} | R2g={r2_g:.4f} | "
              f"lr={optim.param_groups[0]['lr']:.2e} | time={time.time()-t0:.1f}s")

    print(f"\nBest model saved to: {best_path}")
    print(f"Best RMSE (on val): {best_rmse:.6f}")

    ckpt = torch.load(best_path, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])

    test_loss, test_metrics = eval_model(model, dl_te, min_v, max_v)

    # save prediction arrays
    model.eval()
    all_pred_te, all_true_te = [], []
    with torch.no_grad():
        for X_batch, y_batch in dl_te:
            X_batch = to_device(X_batch)
            y_batch = to_device(y_batch)
            gating_hint = compute_gating_signal(X_batch) if cfg.USE_TGATING else None
            Xn_batch = norm_apply(X_batch, min_v, max_v)
            yp_batch, _ = model(Xn_batch, gating_hint=gating_hint)
            ypd_batch = denorm(yp_batch, min_v, max_v)
            all_pred_te.append(ypd_batch.cpu().numpy())
            all_true_te.append(y_batch.cpu().numpy())

    y_pred_test = np.concatenate(all_pred_te, axis=0)
    y_true_test = np.concatenate(all_true_te, axis=0)

    npy_pred_path = best_path.replace(".pt", "_y_pred_test.npy")
    npy_true_path = best_path.replace(".pt", "_y_true_test.npy")
    np.save(npy_pred_path, y_pred_test)
    np.save(npy_true_path, y_true_test)

    print(f"Saved TEST prediction array to: {npy_pred_path}")
    print(f"Saved TEST ground-truth array to: {npy_true_path}")

    # print metrics
    global_metrics = test_metrics.get("global", {})
    per_channel = test_metrics.get("per_channel", None)

    print("\n=== Final Metrics (Test set) ===")
    print("Global metrics:")
    for k in ["rmse", "mae", "smape", "r2"]:
        if k in global_metrics:
            print(f"  {k.upper()}: {global_metrics[k]:.6f}")

    print("\nPer-channel metrics:")
    if per_channel is not None and all(m in per_channel for m in ["rmse", "mae", "smape", "r2"]):
        for i, (ch_idx, ch_name) in enumerate(zip(cfg.CHANNELS, selected_channel_names)):
            rmse_i  = per_channel["rmse"][i]
            mae_i   = per_channel["mae"][i]
            smape_i = per_channel["smape"][i]
            r2_i    = per_channel["r2"][i]
            print(f"  [{ch_idx}] {ch_name}: RMSE={rmse_i:.6f}, MAE={mae_i:.6f}, SMAPE={smape_i:.6f}, R2={r2_i:.6f}")

    # save results json
    results = {
        "config": cfg.__dict__,
        "channels": cfg.CHANNELS,
        "channel_names": selected_channel_names,
        "best_rmse_val": best_rmse,
        "final_metrics": test_metrics
    }
    results_path = best_path.replace('.pt', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_path}")

if __name__ == "__main__":
    # keep user-style default overrides
    cfg.TRAIN_NPY = ""
    cfg.TEST_NPY  = ""
    cfg.CHANNELS = [0]
    cfg.C = len(cfg.CHANNELS)

    cfg.USE_CROP  = False
    cfg.CROP_MODE = "center"
    cfg.CROP_SIZE = 20
    cfg.CROP_BOX  = None

    cfg.NODE_MODE = True
    cfg.COORDS_NPY = None

    cfg.USE_SFE = True
    cfg.USE_TGATING = True
    cfg.USE_GOT = True
    cfg.USE_MGAT = True

    main()

