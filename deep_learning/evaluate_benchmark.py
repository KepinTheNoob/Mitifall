from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ml_pipeline.feature_extraction import FEATURE_NAMES
from ml_pipeline.train_evaluate import build_models

from deep_learning import config
from deep_learning.data import InputLayout, build_windows
from deep_learning.metrics import confusion, metrics_from_counts, select_threshold, threshold_free
from deep_learning.model import count_parameters, estimate_macs
from deep_learning.train import fit_ensemble, predict_proba

SOFT_FALL_QUANTILE = 1.0 / 3.0
BOOTSTRAP_ITERATIONS = 2000
TABLE_COLUMNS = ("accuracy", "f1_macro", "f1_fall", "sensitivity", "specificity",
                 "roc_auc", "pr_auc", "soft_fall_sensitivity")
COLUMN_TITLES = {
    "accuracy": "Accuracy", "f1_macro": "F1 macro", "f1_fall": "F1 Fall",
    "sensitivity": "Sensitivity", "specificity": "Specificity", "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC", "soft_fall_sensitivity": "Soft-fall sens.",
}

def validation_subjects(test_subject: int, fold_index: int) -> tuple[int, ...]:
    pool = [s for s in config.VALIDATION_POOL if s != test_subject]
    start = (fold_index * config.VALIDATION_SUBJECTS_PER_FOLD) % len(pool)
    return tuple(pool[(start + k) % len(pool)] for k in range(config.VALIDATION_SUBJECTS_PER_FOLD))

def soft_fall_mask(table: pd.DataFrame) -> tuple[np.ndarray, float]:
    falls = table[table["label"] == 1]
    peak = falls.groupby("source_file")["acc_mag_max"].max()
    cutoff = float(peak.quantile(SOFT_FALL_QUANTILE))
    soft = set(peak[peak <= cutoff].index)
    rec_peak = table["source_file"].map(peak)
    mask = (table["label"].eq(1) & table["source_file"].isin(soft)
            & np.isclose(table["acc_mag_max"], rec_peak)).to_numpy()
    return mask, cutoff

def run_tag(args: argparse.Namespace) -> str:
    payload = json.dumps({"arch": args.arch, "gyro": args.with_gyro, "baro": args.with_baro,
                          "loss": args.loss, "aug": not args.no_augment, "epochs": args.epochs,
                          "model": asdict(config.ModelConfig(arch=args.arch)),
                          "train": asdict(config.TrainConfig()),
                          "augment": asdict(config.AugmentConfig()),
                          "seed": args.seed,

                          **({"ensemble": args.ensemble} if args.ensemble > 1 else {})},
                         sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    name = args.arch + ("_gyro" if args.with_gyro else "") + ("_baro" if args.with_baro else "")
    name += f"_ens{args.ensemble}" if args.ensemble > 1 else ""
    return f"{name}_{digest}"

def fold_file(out_dir: Path, subject: int) -> Path:
    return out_dir / "folds" / f"subject_{subject:02d}.csv"

def run_folds(args, windows, layout, out_dir: Path) -> None:
    table = windows.table
    y, subjects = windows.labels, windows.subjects
    raw = windows.raw(layout)
    X_rf = table[FEATURE_NAMES].to_numpy(dtype=float)
    (out_dir / "folds").mkdir(parents=True, exist_ok=True)

    all_subjects = sorted(int(s) for s in np.unique(subjects))
    selected = all_subjects if not args.folds else [int(s) for s in args.folds.split(",")]
    mcfg = config.ModelConfig(arch=args.arch)
    tcfg = config.TrainConfig(epochs=args.epochs, loss=args.loss, threads=args.threads)
    acfg = config.AugmentConfig(enabled=not args.no_augment)

    for fold_index, test_subject in enumerate(all_subjects):
        if test_subject not in selected or fold_file(out_dir, test_subject).is_file():
            continue
        val_subjects = validation_subjects(test_subject, fold_index)
        te = subjects == test_subject
        va = np.isin(subjects, val_subjects)
        tr = ~te & ~va

        started = time.perf_counter()
        rf = build_models(("random_forest",), n_jobs=-1)["random_forest"]
        rf.fit(X_rf[tr], y[tr])
        rf_val = rf.predict_proba(X_rf[va])[:, 1]
        rf_test = rf.predict_proba(X_rf[te])[:, 1]
        rf_thr = select_threshold(y[va], rf_val)

        dl, info = fit_ensemble(raw[tr], y[tr], raw[va], y[va], layout, mcfg, tcfg, acfg,
                                seed=args.seed + fold_index, size=args.ensemble)
        dl_test = predict_proba(dl, raw[te])

        fold = pd.DataFrame({
            "row": np.flatnonzero(te), "subject_id": test_subject, "label": y[te],
            "p_rf": rf_test, "p_dl": dl_test, "thr_rf": rf_thr, "thr_dl": info["val_threshold"],
            "dl_best_epoch": info["best_epoch"], "dl_epochs_run": info["epochs_run"],
            "dl_seconds": info["seconds"], "val_subjects": ",".join(map(str, val_subjects)),
        })
        tmp = fold_file(out_dir, test_subject).with_suffix(".tmp")
        fold.to_csv(tmp, index=False)
        tmp.replace(fold_file(out_dir, test_subject))

        def brief(p, thr):
            m = metrics_from_counts(confusion(y[te], p >= thr))
            return f"sens={m['sensitivity']:.2f} spec={m['specificity']:.2f}"

        print(f"[fold {fold_index + 1:2d}/19] subject {test_subject:2d} val={val_subjects} "
              f"n={int(te.sum()):4d} falls={int(y[te].sum()):3d} | "
              f"RF thr={rf_thr:.2f} {brief(rf_test, rf_thr)} | "
              f"DL thr={info['val_threshold']:.2f} {brief(dl_test, info['val_threshold'])} "
              f"ep={info['best_epoch']}/{info['epochs_run']} | {time.perf_counter() - started:.0f}s",
              flush=True)

def run_parallel(args, all_subjects: list[int], out_dir: Path) -> None:
    pending = [s for s in all_subjects if not fold_file(out_dir, s).is_file()]
    if args.folds:
        wanted = {int(s) for s in args.folds.split(",")}
        pending = [s for s in pending if s in wanted]
    if not pending:
        return
    buckets = [pending[i::args.jobs] for i in range(args.jobs)]
    passthrough = [f"--arch={args.arch}", f"--loss={args.loss}", f"--epochs={args.epochs}",
                   f"--seed={args.seed}", f"--threads={args.threads}",
                   f"--ensemble={args.ensemble}", "--worker"]
    passthrough += ["--with-gyro"] if args.with_gyro else []
    passthrough += ["--with-baro"] if args.with_baro else []
    passthrough += ["--no-augment"] if args.no_augment else []
    procs = [subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                               "--folds=" + ",".join(map(str, b)), *passthrough])
             for b in buckets if b]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"worker failure, exit codes {codes}")

def model_rows(pred: pd.DataFrame, arch_label: str) -> dict[str, np.ndarray]:
    return {
        "ml_inference RF @ 0.30 (deployed)": (pred["p_rf"] >= config.ML_INFERENCE_THRESHOLD).to_numpy(),
        "ml_inference RF @ val-tuned": (pred["p_rf"] >= pred["thr_rf"]).to_numpy(),
        f"deep_learning {arch_label} @ val-tuned": (pred["p_dl"] >= pred["thr_dl"]).to_numpy(),
    }

def score(pred: pd.DataFrame, soft: np.ndarray, arch_label: str) -> pd.DataFrame:
    y = pred["label"].to_numpy()
    rows = []
    for name, decision in model_rows(pred, arch_label).items():
        proba = pred["p_dl" if name.startswith("deep") else "p_rf"].to_numpy()
        m = metrics_from_counts(confusion(y, decision))
        m.update(threshold_free(y, proba))
        m["soft_fall_sensitivity"] = float(decision[soft].mean()) if soft.any() else float("nan")
        m.update(dict(zip(("tp", "fp", "tn", "fn"), confusion(y, decision).tolist())))
        rows.append({"model": name, **m})
    return pd.DataFrame(rows)

def bootstrap(pred: pd.DataFrame, soft: np.ndarray, arch_label: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = pred["label"].to_numpy()
    subj = pred["subject_id"].to_numpy()
    groups = {s: np.flatnonzero(subj == s) for s in np.unique(subj)}
    keys = list(groups)
    decisions = model_rows(pred, arch_label)
    names = list(decisions)
    dl_name = names[2]

    deltas = {(base, col): [] for base in names[:2] for col in TABLE_COLUMNS}
    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        yi = y[idx]
        if yi.min() == yi.max():
            continue
        values = {}
        for name in names:
            d = decisions[name][idx]
            proba = pred["p_dl" if name.startswith("deep") else "p_rf"].to_numpy()[idx]
            m = metrics_from_counts(confusion(yi, d))
            m.update(threshold_free(yi, proba))
            s_idx = soft[idx]
            m["soft_fall_sensitivity"] = float(d[s_idx].mean()) if s_idx.any() else float("nan")
            values[name] = m
        for base in names[:2]:
            for col in TABLE_COLUMNS:
                deltas[(base, col)].append(values[dl_name][col] - values[base][col])

    rows = []
    for (base, col), v in deltas.items():
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        verdict = "DL higher" if lo > 0 else ("RF higher" if hi < 0 else "no clear difference")
        rows.append({"baseline": base, "metric": col, "delta_mean": float(v.mean()),
                     "ci_low": float(lo), "ci_high": float(hi), "verdict": verdict})
    return pd.DataFrame(rows)

def _sens_at_spec(y: np.ndarray, p: np.ndarray, spec: float) -> float:
    neg = np.sort(p[y == 0])
    need = int(np.ceil(spec * len(neg) - 1e-9))
    if need <= 0:
        return 1.0
    return float(np.mean(p[y == 1] > neg[min(need, len(neg)) - 1]))

def _spec_at_sens(y: np.ndarray, p: np.ndarray, sens: float) -> float:
    pos = np.sort(p[y == 1])[::-1]
    need = int(np.ceil(sens * len(pos) - 1e-9))
    if need <= 0:
        return 1.0
    return float(np.mean(p[y == 0] < pos[min(need, len(pos)) - 1]))

def matched_points(pred: pd.DataFrame, arch_label: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = pred["label"].to_numpy()
    p_dl = pred["p_dl"].to_numpy()
    subj = pred["subject_id"].to_numpy()
    groups = {s: np.flatnonzero(subj == s) for s in np.unique(subj)}
    keys = list(groups)
    rf_rows = {k: v for k, v in model_rows(pred, arch_label).items() if k.startswith("ml_inference")}

    def point(idx, decision):
        yi, pi = y[idx], p_dl[idx]
        m = metrics_from_counts(confusion(yi, decision[idx]))
        return (m["sensitivity"], _sens_at_spec(yi, pi, m["specificity"]),
                m["specificity"], _spec_at_sens(yi, pi, m["sensitivity"]))

    rows = []
    everything = np.arange(len(y))
    for name, decision in rf_rows.items():
        rf_sens, dl_sens, rf_spec, dl_spec = point(everything, decision)
        d_sens, d_spec = [], []
        for _ in range(BOOTSTRAP_ITERATIONS):
            idx = np.concatenate([groups[k] for k in rng.choice(keys, size=len(keys), replace=True)])
            if y[idx].min() == y[idx].max():
                continue
            a, b, c, d = point(idx, decision)
            d_sens.append(b - a)
            d_spec.append(d - c)
        for metric, rf_v, dl_v, deltas in (("sensitivity at equal specificity", rf_sens, dl_sens, d_sens),
                                           ("specificity at equal sensitivity", rf_spec, dl_spec, d_spec)):
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            verdict = "DL higher" if lo > 0 else ("RF higher" if hi < 0 else "no clear difference")
            rows.append({"baseline": name, "metric": metric, "rf": rf_v, "dl": dl_v,
                         "delta": dl_v - rf_v, "ci_low": float(lo), "ci_high": float(hi),
                         "verdict": verdict})
    return pd.DataFrame(rows)

def markdown_table(frame: pd.DataFrame, columns, titles) -> str:
    head = "| " + " | ".join(titles) + " |"
    sep = "|" + "|".join(["---"] + [":---:"] * (len(titles) - 1)) + "|"
    lines = [head, sep]
    for _, r in frame.iterrows():
        cells = [str(r[columns[0]])] + [f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])
                                        for c in columns[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Head-to-head benchmark: ml_inference RF vs deep model.")
    p.add_argument("--arch", choices=("tcn", "cnn_bilstm"), default="tcn")
    p.add_argument("--with-gyro", action="store_true", help="ablation: add gyroscope channels")
    p.add_argument("--with-baro", action="store_true", help="needs a dataset with pressure data")
    p.add_argument("--loss", choices=("focal", "weighted_bce"), default="focal")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--epochs", type=int, default=config.TrainConfig.epochs)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--ensemble", type=int, default=config.ENSEMBLE_SIZE,
                   help="seeded members per fold, probabilities averaged")
    p.add_argument("--folds", default="", help="comma-separated test subjects (smoke tests only)")
    p.add_argument("--fresh", action="store_true", help="discard checkpointed folds")
    p.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    p.add_argument("--threads", type=int, default=config.TrainConfig.threads,
                   help="torch threads per process")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    layout = InputLayout(use_gyro=args.with_gyro, use_baro=args.with_baro)
    tag = run_tag(args)
    out_dir = config.OUTPUT_DIR / "benchmark" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh and not args.worker:
        for f in (out_dir / "folds").glob("subject_*.csv"):
            f.unlink()

    windows = build_windows(with_baro=args.with_baro, verbose=not args.worker)
    if args.worker:
        run_folds(args, windows, layout, out_dir)
        return 0

    from deep_learning.model import FallNet
    probe = FallNet(layout, config.ModelConfig(arch=args.arch))
    per_member = (f"params={count_parameters(probe)}, "
                  f"MACs/window={estimate_macs(probe, len(layout.raw_inputs), config.WINDOW_SAMPLES)}")
    print(f"benchmark {tag}: {len(windows.table)} windows, channels={layout.model_channels}, "
          + (per_member if args.ensemble == 1 else f"{args.ensemble} members x ({per_member})"),
          flush=True)

    all_subjects = sorted(int(s) for s in np.unique(windows.subjects))
    if args.jobs > 1:
        run_parallel(args, all_subjects, out_dir)
    else:
        run_folds(args, windows, layout, out_dir)

    files = sorted((out_dir / "folds").glob("subject_*.csv"))
    if not files:
        raise SystemExit("no folds completed")
    pred = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).sort_values("row")
    pred = pred.reset_index(drop=True)
    n_subjects = pred["subject_id"].nunique()
    if n_subjects < 19:
        print(f"\nonly {n_subjects}/19 folds available - partial result, not a valid benchmark")

    soft_all, cutoff = soft_fall_mask(windows.table)
    soft = soft_all[pred["row"].to_numpy()]
    arch_label = args.arch.upper().replace("_", "-") + (" +gyro" if args.with_gyro else "")
    arch_label += f" x{args.ensemble} seeds" if args.ensemble > 1 else ""
    table = score(pred, soft, arch_label)
    deltas = bootstrap(pred, soft, arch_label)
    matched = matched_points(pred, arch_label)

    pred.to_csv(out_dir / "predictions.csv", index=False)
    table.to_csv(out_dir / "metrics.csv", index=False)
    deltas.to_csv(out_dir / "bootstrap_deltas.csv", index=False)
    matched.to_csv(out_dir / "matched_operating_points.csv", index=False)

    titles = ["Model"] + [COLUMN_TITLES[c] for c in TABLE_COLUMNS]
    md = [f"# Benchmark: ml_inference vs deep_learning ({arch_label})", "",
          f"- Windows: {len(pred)} test windows over {n_subjects} LOSO folds "
          f"({int(pred['label'].sum())} Fall / {int((pred['label'] == 0).sum())} ADL)",
          f"- Soft-fall subset: impact windows of fall recordings with peak |a| <= {cutoff:.2f} g "
          f"({int(soft.sum())} windows)",
          f"- Deep model: {args.arch}, channels {', '.join(layout.model_channels)}, "
          f"ensemble size {args.ensemble}, "
          f"median best epoch {int(pred.groupby('subject_id')['dl_best_epoch'].first().median())}",
          "", markdown_table(table, ["model", *TABLE_COLUMNS], titles), "",
          "## DL minus RF, subject-level bootstrap (95 % interval)", ""]
    for base in deltas["baseline"].unique():
        md += [f"vs **{base}**", "", "| Metric | Delta | 95 % interval | Verdict |", "|---|:---:|:---:|---|"]
        for _, r in deltas[deltas["baseline"] == base].iterrows():
            md.append(f"| {COLUMN_TITLES[r['metric']]} | {r['delta_mean']:+.3f} | "
                      f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['verdict']} |")
        md.append("")
    md += ["## Same operating point (ROC curves compared at the forest's threshold)", "",
           "The deep model's threshold is read off the pooled test ROC curve to match the "
           "forest's specificity (or sensitivity). A curve comparison like ROC-AUC, not a "
           "deployable threshold.", "",
           "| Forest configuration | Metric | RF | DL | Delta | 95 % interval | Verdict |",
           "|---|---|:---:|:---:|:---:|:---:|---|"]
    for _, r in matched.iterrows():
        md.append(f"| {r['baseline']} | {r['metric']} | {r['rf']:.3f} | {r['dl']:.3f} | "
                  f"{r['delta']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['verdict']} |")
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n" + table[["model", *TABLE_COLUMNS]].round(4).to_string(index=False))
        print("\n" + deltas.round(4).to_string(index=False))
        print("\n" + matched.round(4).to_string(index=False))
    print(f"\nreport: {out_dir / 'report.md'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
