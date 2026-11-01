from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn

from deep_learning import config
from deep_learning.augment import augment_batch
from deep_learning.data import InputLayout, WindowSet, build_windows
from deep_learning.metrics import confusion, metrics_from_counts, select_threshold, threshold_free
from deep_learning.model import (BinaryFocalLoss, FallEnsemble, FallNet, InputPrep, count_parameters,
                                 estimate_macs)

def robust_scaler(raw: np.ndarray, layout: InputLayout) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        derived = InputPrep(layout).derive(torch.from_numpy(raw.transpose(0, 2, 1))).numpy()
    flat = derived.transpose(1, 0, 2).reshape(derived.shape[1], -1)
    center = np.median(flat, axis=1)
    q75, q25 = np.percentile(flat, [75, 25], axis=1)
    scale = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)

@torch.no_grad()
def predict_proba(model: nn.Module, raw: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(raw), batch):
        xb = torch.from_numpy(np.ascontiguousarray(raw[i:i + batch].transpose(0, 2, 1)))
        out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

def _loss_fn(y: np.ndarray, tcfg: config.TrainConfig) -> nn.Module:
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - n_pos, 1)
    if tcfg.loss == "focal":
        return BinaryFocalLoss(alpha=n_neg / (n_pos + n_neg), gamma=tcfg.focal_gamma)
    if tcfg.loss == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos))
    raise ValueError(f"unknown loss {tcfg.loss!r}")

def fit(raw_tr: np.ndarray, y_tr: np.ndarray, raw_va: np.ndarray, y_va: np.ndarray,
        layout: InputLayout, mcfg: config.ModelConfig, tcfg: config.TrainConfig,
        acfg: config.AugmentConfig, seed: int, log=None) -> tuple[FallNet, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(tcfg.threads)

    center, scale = robust_scaler(raw_tr, layout)
    model = FallNet(layout, mcfg, center, scale)
    loss_fn = _loss_fn(y_tr, tcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=tcfg.lr_factor, patience=tcfg.lr_patience, min_lr=tcfg.min_lr)

    y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
    x_va = torch.from_numpy(np.ascontiguousarray(raw_va.transpose(0, 2, 1)))
    y_va_t = torch.from_numpy(y_va.astype(np.float32))

    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        order = rng.permutation(len(raw_tr))
        running, seen = 0.0, 0
        for i in range(0, len(order), tcfg.batch_size):
            idx = order[i:i + tcfg.batch_size]
            if len(idx) < 16:
                continue
            xb = augment_batch(raw_tr[idx], rng, acfg, layout)
            xb = torch.from_numpy(np.ascontiguousarray(xb.transpose(0, 2, 1)))
            loss = loss_fn(model(xb), y_tr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(idx)
            seen += len(idx)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_va), y_va_t))
        sched.step(val_loss)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": val_loss,
                        "lr": opt.param_groups[0]["lr"]})
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, bad = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
        if log is not None and (epoch % 10 == 0 or bad == 0):
            log(f"    epoch {epoch:3d} train {history[-1]['train_loss']:.4f} "
                f"val {val_loss:.4f} lr {opt.param_groups[0]['lr']:.1e}{'  *' if bad == 0 else ''}")
        if bad >= tcfg.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    val_proba = predict_proba(model, raw_va)
    info = {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_loss": best_loss,
        "val_threshold": select_threshold(y_va, val_proba),
        "val_proba": val_proba,
        "seconds": time.perf_counter() - started,
        "history": history,
    }
    return model, info

def member_seeds(seed: int, size: int) -> list[int]:
    return [seed + 1000 * k for k in range(size)]

def fit_ensemble(raw_tr: np.ndarray, y_tr: np.ndarray, raw_va: np.ndarray, y_va: np.ndarray,
                 layout: InputLayout, mcfg: config.ModelConfig, tcfg: config.TrainConfig,
                 acfg: config.AugmentConfig, seed: int, size: int = 1,
                 log=None) -> tuple[FallNet | FallEnsemble, dict]:
    if size == 1:
        return fit(raw_tr, y_tr, raw_va, y_va, layout, mcfg, tcfg, acfg, seed, log)
    members, infos = [], []
    for k, member_seed in enumerate(member_seeds(seed, size)):
        if log is not None:
            log(f"  member {k + 1}/{size} (seed {member_seed})")
        model, info = fit(raw_tr, y_tr, raw_va, y_va, layout, mcfg, tcfg, acfg, member_seed, log)
        members.append(model)
        infos.append(info)
    ensemble = FallEnsemble(members).eval()
    val_proba = predict_proba(ensemble, raw_va)
    info = {
        "best_epoch": int(np.median([i["best_epoch"] for i in infos])),
        "epochs_run": int(sum(i["epochs_run"] for i in infos)),
        "best_val_loss": float(np.mean([i["best_val_loss"] for i in infos])),
        "val_threshold": select_threshold(y_va, val_proba),
        "val_proba": val_proba,
        "seconds": float(sum(i["seconds"] for i in infos)),
        "history": [i["history"] for i in infos],
        "member_best_epochs": [i["best_epoch"] for i in infos],
    }
    return ensemble, info

def save_checkpoint(path: Path, model: FallNet | FallEnsemble, meta: dict) -> None:
    members = model.members if isinstance(model, FallEnsemble) else [model]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"members": [m.state_dict() for m in members], "meta": meta}, path)

def load_checkpoint(path: Path) -> tuple[FallNet | FallEnsemble, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    layout = InputLayout(use_gyro=meta["use_gyro"], use_baro=meta["use_baro"])
    mcfg = config.ModelConfig(**meta["model_config"])
    members = []
    for state in ckpt["members"]:
        net = FallNet(layout, mcfg)
        net.load_state_dict(state)
        members.append(net.eval())
    model = members[0] if len(members) == 1 else FallEnsemble(members).eval()
    return model, meta

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the final deep-learning fall detector.")
    p.add_argument("--arch", choices=("tcn", "cnn_bilstm"), default="tcn")
    p.add_argument("--with-gyro", action="store_true", help="ablation: add gyroscope channels")
    p.add_argument("--with-baro", action="store_true",
                   help="add the barometric dh channel (needs a dataset with pressure data)")
    p.add_argument("--loss", choices=("focal", "weighted_bce"), default="focal")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--epochs", type=int, default=config.TrainConfig.epochs)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--ensemble", type=int, default=config.ENSEMBLE_SIZE,
                   help="independently seeded members, probabilities averaged")
    p.add_argument("--threads", type=int, default=config.TrainConfig.threads)
    p.add_argument("--out", type=Path, default=config.FINAL_CHECKPOINT)
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    layout = InputLayout(use_gyro=args.with_gyro, use_baro=args.with_baro)
    mcfg = config.ModelConfig(arch=args.arch)
    tcfg = config.TrainConfig(epochs=args.epochs, loss=args.loss, threads=args.threads)
    acfg = config.AugmentConfig(enabled=not args.no_augment)

    windows: WindowSet = build_windows(with_baro=args.with_baro)
    y, subjects = windows.labels, windows.subjects
    raw = windows.raw(layout)
    val_mask = np.isin(subjects, config.FINAL_VALIDATION_SUBJECTS)
    print(f"final model: arch={mcfg.arch} channels={layout.model_channels}")
    print(f"  train {int((~val_mask).sum())} windows / val {int(val_mask.sum())} windows "
          f"(subjects {config.FINAL_VALIDATION_SUBJECTS})")

    model, info = fit_ensemble(raw[~val_mask], y[~val_mask], raw[val_mask], y[val_mask],
                               layout, mcfg, tcfg, acfg, args.seed, args.ensemble, log=print)
    counts = confusion(y[val_mask], info["val_proba"] >= info["val_threshold"])
    val_metrics = {**metrics_from_counts(counts), **threshold_free(y[val_mask], info["val_proba"])}

    first = model.members[0] if isinstance(model, FallEnsemble) else model
    meta = {
        "arch": mcfg.arch,
        "ensemble": args.ensemble,
        "seeds": member_seeds(args.seed, args.ensemble),
        "model_config": asdict(mcfg),
        "train_config": asdict(tcfg),
        "augment_config": asdict(acfg),
        "raw_inputs": list(layout.raw_inputs),
        "model_channels": list(layout.model_channels),
        "use_gyro": layout.use_gyro,
        "use_baro": layout.use_baro,
        "window_samples": config.WINDOW_SAMPLES,
        "hop_samples": config.HOP_SAMPLES,
        "sample_rate_hz": config.SAMPLE_RATE_HZ,
        "lowpass_cutoff_hz": config.LOWPASS_CUTOFF_HZ,
        "lowpass_order": config.LOWPASS_ORDER,
        "threshold": round(float(info["val_threshold"]), 4),
        "best_epoch": info["best_epoch"],
        "validation_subjects": list(config.FINAL_VALIDATION_SUBJECTS),
        "validation_metrics": val_metrics,
        "parameters": count_parameters(model),
        "macs_per_window": estimate_macs(model, len(layout.raw_inputs), config.WINDOW_SAMPLES),

        "scaler_center": first.prep.center.flatten().tolist(),
        "scaler_scale": first.prep.scale.flatten().tolist(),
    }
    save_checkpoint(args.out, model, meta)
    print(f"\nsaved {args.out}")
    print(json.dumps({k: meta[k] for k in ("threshold", "best_epoch", "parameters", "macs_per_window")}))
    print("validation (subjects " + ", ".join(map(str, config.FINAL_VALIDATION_SUBJECTS)) + "): "
          + " ".join(f"{k}={v:.3f}" for k, v in val_metrics.items()))
    return 0

if __name__ == "__main__":
    sys.exit(main())
