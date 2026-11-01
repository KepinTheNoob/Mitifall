from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from ml_pipeline import config as ml_config
from ml_pipeline.feature_extraction import FEATURE_NAMES
from ml_pipeline.train_evaluate import build_models

from deep_learning import config
from deep_learning.data import InputLayout, load_ml_feature_table
from deep_learning.evaluate_benchmark import (_sens_at_spec, _spec_at_sens, run_tag,
                                              soft_fall_mask, validation_subjects)
from deep_learning.evaluate_benchmark import parse_args as benchmark_args
from deep_learning.metrics import select_threshold
from deep_learning.model import FallEnsemble, FallNet, count_parameters, estimate_macs

OUT_DIR = config.OUTPUT_DIR / "evaluation"
FIG_DIR = OUT_DIR / "figures"
ML_CACHE = OUT_DIR / "ml_predictions.csv"
ML_CACHE_META = OUT_DIR / "ml_predictions.json"
BENCH_DIR = config.OUTPUT_DIR / "benchmark"
RF_HEADER = config.PROJECT_ROOT / "include" / "RandomForest.h"

BOOTSTRAP_ITERATIONS = 2000
COOLDOWN_WINDOWS = 6
SENS_LEVELS = (0.80, 0.90, 0.95)
SPEC_LEVELS = (0.90, 0.95)
CI_METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "f1_fall", "sensitivity",
              "specificity", "precision", "mcc", "roc_auc", "pr_auc", "impact_sensitivity",
              "soft_fall_sensitivity")
SUMMARY_METRICS = ("accuracy", "f1_macro", "f1_fall", "sensitivity", "specificity", "mcc",
                   "roc_auc", "pr_auc", "impact_sensitivity", "soft_fall_sensitivity")
TITLES = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.", "f1_macro": "F1 macro",
          "f1_fall": "F1 Fall", "sensitivity": "Sensitivity", "specificity": "Specificity",
          "precision": "Precision", "mcc": "MCC", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
          "brier": "Brier", "impact_sensitivity": "Impact sens.",
          "soft_fall_sensitivity": "Soft-fall sens.",
          "false_alarms_per_hour": "False alarms / h"}

RULE_BASED_REFERENCE = {"sensitivity": 0.856, "specificity": 0.942, "tp": 178, "fn": 30,
                        "fp": 31, "tn": 507}

SENSOR_SUBSETS = {"all9": ml_config.SENSOR_AXES,
                  "acc_gyro": ml_config.ACC_AXES + ml_config.GYR_AXES,
                  "acc": ml_config.ACC_AXES}
SUBSET_INPUTS = {"all9": "acc + gyro + mag", "acc_gyro": "acc + gyro", "acc": "acc"}

@dataclass(frozen=True)
class MLSpec:
    key: str
    short: str
    model: str
    subset: str

ML_SPECS = (
    MLSpec("rf53", "RF-53", "random_forest", "all9"),
    MLSpec("rf38", "RF-38", "random_forest", "acc_gyro"),
    MLSpec("rf19", "RF-19", "random_forest", "acc"),
    MLSpec("svm_rbf", "SVM-RBF", "svm_rbf", "all9"),
    MLSpec("svm_linear", "Linear SVM", "svm_linear", "all9"),
    MLSpec("logreg", "LogReg", "logistic_regression", "all9"),
)

def feature_subset(subset: str) -> list[str]:
    axes = SENSOR_SUBSETS[subset]
    keep = []
    for name in FEATURE_NAMES:
        prefix = name.split("_")[0]
        if prefix in ml_config.SENSOR_AXES:
            if prefix in axes:
                keep.append(name)
        elif prefix == "acc" or (prefix == "gyr" and ml_config.GYR_AXES[0] in axes):
            keep.append(name)
    return keep

@dataclass
class Config:

    key: str
    label: str
    family: str
    source: str
    score: np.ndarray
    threshold: np.ndarray
    inputs: str
    probabilistic: bool = True
    role: str = ""

    @property
    def decision(self) -> np.ndarray:
        return self.score >= self.threshold

@dataclass
class DLRun:

    tag: str
    arch: str
    gyro: bool
    baro: bool
    ensemble: int
    pred: pd.DataFrame

    @property
    def layout(self) -> InputLayout:
        return InputLayout(use_gyro=self.gyro, use_baro=self.baro)

    @property
    def short(self) -> str:
        name = "TCN" if self.arch == "tcn" else "CNN-BiLSTM"
        name += (" +gyro" if self.gyro else "") + (" +baro" if self.baro else "")
        return name + (f" x{self.ensemble}" if self.ensemble > 1 else "")

    @property
    def inputs(self) -> str:
        return "acc" + (" + gyro" if self.gyro else "") + (" + baro" if self.baro else "")

TAG_RE = re.compile(r"^(tcn|cnn_bilstm)(_gyro)?(_baro)?(?:_ens(\d+))?_[0-9a-f]{8}$")

def load_dl_runs(table: pd.DataFrame) -> list[DLRun]:
    runs = []
    for path in sorted(BENCH_DIR.glob("*/predictions.csv")):
        m = TAG_RE.match(path.parent.name)
        if m is None:
            continue
        pred = pd.read_csv(path).sort_values("row").reset_index(drop=True)
        if not np.array_equal(pred["row"].to_numpy(), np.arange(len(table))):
            print(f"skipping {path.parent.name}: not all {len(table)} windows scored")
            continue
        if not (np.array_equal(pred["label"], table["label"])
                and np.array_equal(pred["subject_id"], table["subject_id"])):
            raise SystemExit(f"{path.parent.name}: rows do not match the feature table")
        runs.append(DLRun(path.parent.name, m.group(1), bool(m.group(2)), bool(m.group(3)),
                          int(m.group(4) or 1), pred))
    if not runs:
        raise SystemExit("no finished benchmark run - run deep_learning/evaluate_benchmark.py")
    return runs

def fold_plan(subjects: np.ndarray) -> list[tuple[int, int, tuple[int, ...]]]:
    ordered = sorted(int(s) for s in np.unique(subjects))
    return [(i, s, validation_subjects(s, i)) for i, s in enumerate(ordered)]

def _fit_fold(spec: MLSpec, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, fold) -> dict:
    from ml_pipeline.export_model import FlatForest

    _, test, val = fold
    te, va = subjects == test, np.isin(subjects, val)
    tr = ~te & ~va
    Xs = X[:, [FEATURE_NAMES.index(f) for f in feature_subset(spec.subset)]]
    model = build_models((spec.model,), n_jobs=1)[spec.model]
    started = time.perf_counter()
    model.fit(Xs[tr], y[tr])
    fit_s = time.perf_counter() - started

    def score(rows: np.ndarray) -> np.ndarray:
        if spec.model.startswith("svm"):
            return 1.0 / (1.0 + np.exp(-model.decision_function(Xs[rows])))
        return model.predict_proba(Xs[rows])[:, 1]

    started = time.perf_counter()
    p_test = score(te)
    infer_us = (time.perf_counter() - started) / te.sum() * 1e6
    out = {"key": spec.key, "rows": np.flatnonzero(te), "p": p_test,
           "thr": select_threshold(y[va], score(va)), "fit_s": fit_s, "infer_us": infer_us}
    clf = model.named_steps["clf"]
    if spec.model == "random_forest":
        visited, _ = clf.decision_path(Xs[te])
        flat = FlatForest(clf)
        out.update(node_tests=float(np.asarray(visited.sum(axis=1)).mean()) - clf.n_estimators,
                   nodes=flat.n_nodes, flash_kib=flat.flash_bytes() / 1024.0)
    elif spec.model.startswith("svm"):
        out["support_vectors"] = int(clf.support_vectors_.shape[0])
    return out

def ml_cache_key(table: pd.DataFrame) -> str:
    import sklearn

    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(table[["source_file", "window_start_ms", "label"]],
                                        index=False).values.tobytes())
    h.update(np.ascontiguousarray(table[FEATURE_NAMES].to_numpy(float)).tobytes())
    h.update(repr((ML_SPECS, ml_config.RF_PARAMS, ml_config.SVM_RBF_PARAMS,
                   ml_config.SVM_LINEAR_PARAMS, ml_config.LOGREG_PARAMS, config.VALIDATION_POOL,
                   config.VALIDATION_SUBJECTS_PER_FOLD, sklearn.__version__)).encode())
    return h.hexdigest()[:16]

def classic_predictions(table: pd.DataFrame, refit: bool, jobs: int) -> tuple[pd.DataFrame, dict]:
    key = ml_cache_key(table)
    if not refit and ML_CACHE.is_file() and ML_CACHE_META.is_file():
        meta = json.loads(ML_CACHE_META.read_text(encoding="utf-8"))
        if meta.get("key") == key:
            print(f"classic models: cached predictions ({ML_CACHE.name})")
            return pd.read_csv(ML_CACHE), meta

    from joblib import Parallel, delayed

    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(config.PROJECT_ROOT), os.environ.get("PYTHONPATH")) if p)
    X = table[FEATURE_NAMES].to_numpy(float)
    y = table["label"].to_numpy(int)
    subjects = table["subject_id"].to_numpy(int)
    tasks = [(spec, fold) for spec in ML_SPECS for fold in fold_plan(subjects)]
    print(f"classic models: refitting {len(tasks)} model-folds on {jobs} workers", flush=True)
    started = time.perf_counter()
    results = Parallel(n_jobs=jobs)(delayed(_fit_fold)(s, X, y, subjects, f) for s, f in tasks)
    frame = pd.DataFrame({"row": np.arange(len(table)), "subject_id": subjects, "label": y})
    meta = {"key": key, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(time.perf_counter() - started, 1), "workers": jobs, "models": {}}
    for spec in ML_SPECS:
        p, thr, extras = np.full(len(table), np.nan), np.full(len(table), np.nan), {}
        for r in (r for r in results if r["key"] == spec.key):
            p[r["rows"]], thr[r["rows"]] = r["p"], r["thr"]
            for name, value in r.items():
                if name not in ("key", "rows", "p", "thr"):
                    extras.setdefault(name, []).append(value)
        frame[f"p_{spec.key}"], frame[f"thr_{spec.key}"] = p, thr
        meta["models"][spec.key] = {k: float(np.median(v)) for k, v in extras.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ML_CACHE, index=False)
    ML_CACHE_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return frame, meta

def shipped_tag() -> str | None:
    if not config.MODEL_CARD.is_file():
        return None
    card = json.loads(config.MODEL_CARD.read_text(encoding="utf-8"))
    return (card.get("benchmark") or {}).get("tag")

def build_configs(ml: pd.DataFrame, runs: list[DLRun], shipped: str | None,
                  prespecified: str) -> list[Config]:
    rf = ml["p_rf53"].to_numpy()
    configs = [Config("rf53@0.30", "RF-53 @ 0.30", "ML", "rf53", rf,
                      np.full(len(rf), config.ML_INFERENCE_THRESHOLD), SUBSET_INPUTS["all9"],
                      role="deployed in ml_inference")]
    for spec in ML_SPECS:
        configs.append(Config(f"{spec.key}@val", f"{spec.short} @ val", "ML", spec.key,
                              ml[f"p_{spec.key}"].to_numpy(), ml[f"thr_{spec.key}"].to_numpy(),
                              SUBSET_INPUTS[spec.subset],
                              probabilistic=not spec.model.startswith("svm")))
    for run in runs:
        role = ("shipped ONNX, picked after comparing" if run.tag == shipped
                else "pre-specified" if run.tag == prespecified else "")
        configs.append(Config(run.tag, f"{run.short} @ val", "DL", run.tag,
                              run.pred["p_dl"].to_numpy(), run.pred["thr_dl"].to_numpy(),
                              run.inputs, role=role))
    return configs

def rf_consistency(ml: pd.DataFrame, runs: list[DLRun]) -> float:
    worst = 0.0
    for run in runs:
        for col in ("p_rf", "thr_rf"):
            diff = np.abs(run.pred[col].to_numpy() - ml[col.replace("_rf", "_rf53")].to_numpy())
            worst = max(worst, float(diff.max()))
    return worst

def matched_forest(dl: Config, by_key: dict[str, Config]) -> Config | None:
    return {"acc": by_key.get("rf19@val"), "acc + gyro": by_key.get("rf38@val")}.get(dl.inputs)

def counts_by_group(decision: np.ndarray, y: np.ndarray, group: np.ndarray, n: int) -> np.ndarray:
    d, t = decision.astype(bool), y.astype(bool)
    cells = (t & d, ~t & d, ~t & ~d, t & ~d)
    return np.stack([np.bincount(group[c], minlength=n) for c in cells], axis=1)

def count_metrics(c: np.ndarray) -> dict[str, np.ndarray]:
    tp, fp, tn, fn = (c[..., i].astype(float) for i in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        prec, npv = tp / (tp + fp), tn / (tn + fn)
        f1_fall = 2 * prec * sens / (prec + sens)
        f1_adl = 2 * npv * spec / (npv + spec)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return {"accuracy": (tp + tn) / (tp + fp + tn + fn),
                "balanced_accuracy": 0.5 * (sens + spec), "precision": prec,
                "sensitivity": sens, "specificity": spec, "f1_fall": f1_fall,
                "f1_macro": 0.5 * (f1_fall + f1_adl), "mcc": mcc}

def auc(y: np.ndarray, p: np.ndarray) -> float:
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((stats.rankdata(p)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def impact_masks(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    soft, cutoff = soft_fall_mask(table)
    peak = table[table["label"] == 1].groupby("source_file")["acc_mag_max"].max()
    rec_peak = table["source_file"].map(peak)
    impact = (table["label"].eq(1) & np.isclose(table["acc_mag_max"], rec_peak)).to_numpy()
    return impact, soft, cutoff

def verdict(lo: float, hi: float) -> str:
    return "DL higher" if lo > 0 else ("ML higher" if hi < 0 else "no clear difference")

class Bootstrap:

    def __init__(self, subjects: np.ndarray, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = 0):
        self.keys = np.unique(subjects)
        self.index = np.searchsorted(self.keys, subjects)
        rng = np.random.default_rng(seed)
        self.picks = np.stack([rng.choice(len(self.keys), size=len(self.keys), replace=True)
                               for _ in range(iterations)])
        self.rows = [np.flatnonzero(self.index == k) for k in range(len(self.keys))]

    @property
    def n(self) -> int:
        return len(self.keys)

    def resample(self, per_subject: np.ndarray) -> np.ndarray:
        return per_subject[self.picks].sum(axis=1)

    def ratio(self, hits: np.ndarray, total: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.resample(hits) / self.resample(total)

    def window_samples(self, cfg: Config, y: np.ndarray, impact: np.ndarray, soft: np.ndarray,
                       score_cache: dict) -> dict[str, np.ndarray]:
        out = count_metrics(self.resample(counts_by_group(cfg.decision, y, self.index, self.n)))
        for name, mask in (("impact_sensitivity", impact), ("soft_fall_sensitivity", soft)):
            out[name] = self.ratio(np.bincount(self.index[mask & cfg.decision], minlength=self.n),
                                   np.bincount(self.index[mask], minlength=self.n))
        if cfg.source not in score_cache:
            score_cache[cfg.source] = self.score_samples(y, cfg.score)
        out.update(score_cache[cfg.source])
        return out

    def score_samples(self, y: np.ndarray, score: np.ndarray,
                      rows: list[np.ndarray] | None = None) -> dict[str, np.ndarray]:
        rows = self.rows if rows is None else rows
        roc, pr = np.empty(len(self.picks)), np.empty(len(self.picks))
        for b, pick in enumerate(self.picks):
            idx = np.concatenate([rows[k] for k in pick])
            roc[b], pr[b] = auc(y[idx], score[idx]), average_precision_score(y[idx], score[idx])
        return {"roc_auc": roc, "pr_auc": pr}

def window_metrics(cfg: Config, y: np.ndarray, impact: np.ndarray, soft: np.ndarray) -> dict:
    c = counts_by_group(cfg.decision, y, np.zeros(len(y), dtype=int), 1)[0]
    m = {k: float(v) for k, v in count_metrics(c).items()}
    m.update(roc_auc=auc(y, cfg.score), pr_auc=float(average_precision_score(y, cfg.score)),
             brier=float(np.mean((cfg.score - y) ** 2)) if cfg.probabilistic else float("nan"),
             impact_sensitivity=float(cfg.decision[impact].mean()),
             soft_fall_sensitivity=float(cfg.decision[soft].mean()))
    m.update(dict(zip(("tp", "fp", "tn", "fn"), (int(v) for v in c))))
    return m

def delta_table(point: dict, samples: dict, pairs: list[tuple[Config, Config]],
                metrics=CI_METRICS) -> pd.DataFrame:
    rows = []
    for dl, ml in pairs:
        for metric in metrics:
            d = samples[dl.key][metric] - samples[ml.key][metric]
            lo, hi = np.percentile(d[np.isfinite(d)], [2.5, 97.5])
            rows.append({"dl": dl.key, "dl_label": dl.label, "ml": ml.key, "ml_label": ml.label,
                         "metric": metric, "dl_value": point[dl.key][metric],
                         "ml_value": point[ml.key][metric],
                         "delta": point[dl.key][metric] - point[ml.key][metric],
                         "ci_low": float(lo), "ci_high": float(hi), "verdict": verdict(lo, hi)})
    return pd.DataFrame(rows)

def mcnemar(a: Config, b: Config, y: np.ndarray, cluster: np.ndarray, n: int) -> dict:
    truth = y.astype(bool)
    only_a = (a.decision == truth) & (b.decision != truth)
    only_b = (a.decision != truth) & (b.decision == truth)
    na, nb = int(only_a.sum()), int(only_b.sum())
    p_std = stats.binomtest(na, na + nb, 0.5).pvalue if na + nb else 1.0
    ka, kb = np.bincount(cluster[only_a], minlength=n), np.bincount(cluster[only_b], minlength=n)
    den = float(((ka - kb) ** 2).sum())
    chi2 = float((ka.sum() - kb.sum()) ** 2 / den) if den else 0.0
    return {"dl": a.key, "dl_label": a.label, "ml": b.key, "ml_label": b.label,
            "only_dl_correct": na, "only_ml_correct": nb, "p_mcnemar": float(p_std),
            "chi2_clustered": chi2, "p_clustered": float(stats.chi2.sf(chi2, 1))}

def per_subject(configs: list[Config], y: np.ndarray, boot: Bootstrap) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        m = count_metrics(counts_by_group(cfg.decision, y, boot.index, boot.n))
        for k, subject in enumerate(boot.keys):
            r = boot.rows[k]
            rows.append({"config": cfg.key, "label": cfg.label, "subject": int(subject),
                         "windows": len(r), "falls": int(y[r].sum()),
                         **{name: float(v[k]) for name, v in m.items()},
                         "roc_auc": auc(y[r], cfg.score[r])})
    return pd.DataFrame(rows)

def wilcoxon_table(ps: pd.DataFrame, pairs: list[tuple[Config, Config]],
                   metrics=("roc_auc", "balanced_accuracy", "sensitivity", "specificity",
                            "accuracy")) -> pd.DataFrame:
    rows = []
    for dl, ml in pairs:
        a = ps[ps["config"] == dl.key].set_index("subject")
        b = ps[ps["config"] == ml.key].set_index("subject")
        for metric in metrics:
            d = (a[metric] - b[metric]).dropna()
            nonzero = d[d != 0]
            p = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 2 else float("nan")
            rows.append({"dl": dl.key, "dl_label": dl.label, "ml": ml.key, "ml_label": ml.label,
                         "metric": metric, "subjects": len(d), "dl_better": int((d > 0).sum()),
                         "ml_better": int((d < 0).sum()), "ties": int((d == 0).sum()),
                         "median_delta": float(d.median()) if len(d) else float("nan"), "p": p})
    return pd.DataFrame(rows)

def per_activity(configs: list[Config], table: pd.DataFrame, impact: np.ndarray) -> pd.DataFrame:
    rows = []
    for (cls, activity), g in table.groupby(["class_dir", "activity"], sort=True):
        idx = g.index.to_numpy()
        imp = idx[impact[idx]]
        for cfg in configs:
            d = cfg.decision
            rows.append({"class": cls, "activity": activity, "recordings": g["source_file"].nunique(),
                         "windows": len(idx), "config": cfg.key, "label": cfg.label,
                         "positive_rate": float(d[idx].mean()),
                         "impact_detection": float(d[imp].mean()) if len(imp) else float("nan")})
    return pd.DataFrame(rows)

def fixed_points(configs: list[Config], y: np.ndarray) -> pd.DataFrame:
    rows, seen = [], set()
    for cfg in configs:
        if cfg.source in seen:
            continue
        seen.add(cfg.source)
        row = {"source": cfg.source, "model": cfg.label.split(" @")[0], "family": cfg.family}
        for level in SENS_LEVELS:
            row[f"spec_at_sens_{level:.2f}"] = _spec_at_sens(y, cfg.score, level)
        for level in SPEC_LEVELS:
            row[f"sens_at_spec_{level:.2f}"] = _sens_at_spec(y, cfg.score, level)
        rows.append(row)
    return pd.DataFrame(rows)

def fixed_point_deltas(R: SimpleNamespace, pairs: list[tuple[Config, Config]]) -> pd.DataFrame:
    tests = ([("spec_at_sens", v, _spec_at_sens) for v in SENS_LEVELS]
             + [("sens_at_spec", v, _sens_at_spec) for v in SPEC_LEVELS])
    keys = [(dl, ml, kind, level, fn) for dl, ml in pairs for kind, level, fn in tests]
    samples = np.empty((len(R.boot.picks), len(keys)))
    for b, pick in enumerate(R.boot.picks):
        idx = np.concatenate([R.boot.rows[k] for k in pick])
        yi, cache = R.y[idx], {}
        for j, (dl, ml, kind, level, fn) in enumerate(keys):
            for cfg in (dl, ml):
                if (cfg.source, kind, level) not in cache:
                    cache[(cfg.source, kind, level)] = fn(yi, cfg.score[idx], level)
            samples[b, j] = cache[(dl.source, kind, level)] - cache[(ml.source, kind, level)]
    rows = []
    for j, (dl, ml, kind, level, fn) in enumerate(keys):
        lo, hi = np.percentile(samples[:, j], [2.5, 97.5])
        rows.append({"dl": dl.key, "dl_label": dl.label, "ml": ml.key, "ml_label": ml.label,
                     "metric": f"{kind}_{level:.2f}",
                     "delta": fn(R.y, dl.score, level) - fn(R.y, ml.score, level),
                     "ci_low": float(lo), "ci_high": float(hi), "verdict": verdict(lo, hi)})
    return pd.DataFrame(rows)

@dataclass
class Recordings:
    windows: list[np.ndarray]
    label: np.ndarray
    subject: np.ndarray
    hours: np.ndarray

    def rows_by_subject(self, n: int) -> list[np.ndarray]:
        return [np.flatnonzero(self.subject == k) for k in range(n)]

def build_recordings(table: pd.DataFrame, boot: Bootstrap) -> Recordings:
    groups = [g.sort_values("window_start_ms").index.to_numpy()
              for _, g in table.groupby("source_file", sort=True)]
    start, end = table["window_start_ms"].to_numpy(float), table["window_end_ms"].to_numpy(float)
    return Recordings(windows=groups,
                      label=np.array([int(table.at[g[0], "label"]) for g in groups]),
                      subject=np.array([boot.index[g[0]] for g in groups]),
                      hours=np.array([(end[g].max() - start[g].min()) / 3.6e6 for g in groups]))

def alarm_events(decision: np.ndarray, confirm: int) -> int:
    trig = decision if confirm == 1 else decision & np.r_[False, decision[:-1]]
    events, ready = 0, 0
    for k in np.flatnonzero(trig):
        if k >= ready:
            events, ready = events + 1, k + COOLDOWN_WINDOWS
    return events

def recording_scores(score: np.ndarray, recs: Recordings, confirm: int) -> np.ndarray:
    out = np.zeros(len(recs.windows))
    for i, rows in enumerate(recs.windows):
        p = score[rows]
        if confirm == 1:
            out[i] = p.max()
        elif len(p) > 1:
            out[i] = np.minimum(p[1:], p[:-1]).max()
    return out

def event_metrics(cfg: Config, recs: Recordings, confirm: int,
                  boot: Bootstrap) -> tuple[dict, dict[str, np.ndarray]]:
    events = np.array([alarm_events(cfg.decision[r], confirm) for r in recs.windows])
    per_subject = counts_by_group(events > 0, recs.label, recs.subject, boot.n)
    adl = recs.label == 0
    alarms = np.bincount(recs.subject[adl], weights=events[adl], minlength=boot.n)
    hours = np.bincount(recs.subject[adl], weights=recs.hours[adl], minlength=boot.n)
    total = per_subject.sum(axis=0)
    point = {k: float(v) for k, v in count_metrics(total).items()}
    point.update(dict(zip(("tp", "fp", "tn", "fn"), (int(v) for v in total))),
                 false_alarms_per_hour=float(alarms.sum() / hours.sum()),
                 adl_hours=float(hours.sum()))
    samples = count_metrics(boot.resample(per_subject))
    samples["false_alarms_per_hour"] = boot.ratio(alarms, hours)
    return point, samples

EVENT_METRICS = ("sensitivity", "specificity", "balanced_accuracy", "f1_fall",
                 "false_alarms_per_hour")

def event_deltas(points: dict, samples: dict, pairs: list[tuple[Config, Config]]) -> pd.DataFrame:
    rows = []
    for confirm in (1, 2):
        for dl, ml in pairs:
            for metric in EVENT_METRICS:
                d = samples[(dl.key, confirm)][metric] - samples[(ml.key, confirm)][metric]
                lo, hi = np.percentile(d[np.isfinite(d)], [2.5, 97.5])
                v = verdict(lo, hi)
                if metric == "false_alarms_per_hour":
                    v = {"DL higher": "ML fewer", "ML higher": "DL fewer"}.get(v, v)
                rows.append({"confirm": confirm, "dl": dl.key, "dl_label": dl.label, "ml": ml.key,
                             "ml_label": ml.label, "metric": metric,
                             "dl_value": points[(dl.key, confirm)][metric],
                             "ml_value": points[(ml.key, confirm)][metric],
                             "delta": points[(dl.key, confirm)][metric]
                             - points[(ml.key, confirm)][metric],
                             "ci_low": float(lo), "ci_high": float(hi), "verdict": v})
    return pd.DataFrame(rows)

NATIVE_BENCH = r"""
#include <chrono>
#include <cstdio>
#include "RandomForest.h"
#include "feature_extractor.h"
int main() {
    using Clock = std::chrono::steady_clock;
    FeatureExtractor::RingBuffer<100> window;
    for (int i = 0; i < 100; ++i) {
        float s[9];
        for (int a = 0; a < 9; ++a) s[a] = 0.01f * static_cast<float>((i * 7 + a * 13) % 97) - 0.4f;
        window.push(s);
    }
    float features[FeatureExtractor::kFeatureCount];
    volatile float sink = 0.0f;
    const int n = 20000;
    const auto t0 = Clock::now();
    for (int r = 0; r < n; ++r) { FeatureExtractor::compute(window, features); sink = sink + features[r % 53]; }
    const auto t1 = Clock::now();
    for (int r = 0; r < n; ++r)
        sink = sink + RandomForest::predictProba(RandomForest::kSelfTestInput[r % RandomForest::kSelfTestCount]);
    const auto t2 = Clock::now();
    const double per = 1e-3 / n;  // ns -> us per call
    std::printf("%.4f %.4f %d\n", std::chrono::duration<double, std::nano>(t1 - t0).count() * per,
                std::chrono::duration<double, std::nano>(t2 - t1).count() * per,
                RandomForest::selfTest() ? 1 : 0);
    return 0;
}
"""

def native_forest_latency() -> dict | None:
    import shutil

    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        src, exe = Path(tmp) / "bench.cpp", Path(tmp) / "bench"
        src.write_text(NATIVE_BENCH, encoding="utf-8")
        cmd = [compiler, "-O2", "-std=c++17", "-I", str(config.PROJECT_ROOT / "include"),
               str(config.PROJECT_ROOT / "src" / "feature_extractor.cpp"), str(src), "-o", str(exe)]
        if subprocess.run(cmd, capture_output=True, timeout=300).returncode != 0:
            return None
        fields = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300).stdout.split()
    return {"compiler": f"{Path(compiler).name} -O2", "features_us": float(fields[0]),
            "forest_us": float(fields[1]), "selftest": fields[2] == "1"}

def deployed_forest() -> dict:
    text = RF_HEADER.read_text(encoding="utf-8", errors="replace")[:4000] if RF_HEADER.is_file() else ""
    nodes = re.search(r"Nodes\s*:\s*(\d+)", text)
    flash = re.search(r"Flash\s*:\s*~([\d.]+)\s*KiB", text)
    return {"nodes": int(nodes.group(1)) if nodes else None,
            "flash_kib": float(flash.group(1)) if flash else None}

def dl_compute(run: DLRun) -> dict:
    import torch

    torch.set_num_threads(1)
    nets = [FallNet(run.layout, config.ModelConfig(arch=run.arch)).eval() for _ in range(run.ensemble)]
    model = nets[0] if run.ensemble == 1 else FallEnsemble(nets).eval()
    n_raw = len(run.layout.raw_inputs)
    one = torch.randn(1, n_raw, config.WINDOW_SAMPLES)
    batch = torch.randn(256, n_raw, config.WINDOW_SAMPLES)
    times = []
    with torch.no_grad():
        for _ in range(20):
            model(one)
        for _ in range(200):
            started = time.perf_counter()
            model(one)
            times.append(time.perf_counter() - started)
        started = time.perf_counter()
        model(batch)
        batched = (time.perf_counter() - started) / len(batch)
    return {"parameters": count_parameters(model),
            "macs": estimate_macs(nets[0], n_raw, config.WINDOW_SAMPLES) * run.ensemble,
            "torch_ms_batch1": float(np.median(times) * 1e3), "torch_us_batched": batched * 1e6,
            "train_s_per_fold": float(run.pred.groupby("subject_id")["dl_seconds"].first().median())}

ML_COLORS = {"rf53": "#1f77b4", "rf38": "#17becf", "rf19": "#9ecae1", "svm_rbf": "#5f5f5f",
             "svm_linear": "#a0a0a0", "logreg": "#c8c8c8"}
DEPLOYED_COLOR = "#6a3d9a"
DL_COLORS = ("#c00000", "#ed7d31", "#bf9000", "#548235", "#e377c2", "#8c564b")

def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    return plt

def colour(cfg: Config, dl_order: list[str], per_model: bool = False) -> str:
    if cfg.family == "ML":
        if cfg.key == "rf53@0.30" and not per_model:
            return DEPLOYED_COLOR
        return ML_COLORS.get(cfg.source, "#7f7f7f")
    return DL_COLORS[dl_order.index(cfg.source) % len(DL_COLORS)]

def plot_curves(configs: list[Config], y: np.ndarray, point: dict, marked: list[tuple[Config, str]],
                path: Path, dl_order: list[str]) -> None:
    plt = _pyplot()
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(12, 5))
    seen = set()
    for cfg in configs:
        if cfg.source in seen:
            continue
        seen.add(cfg.source)
        style = {"color": colour(cfg, dl_order, per_model=True),
                 "lw": 1.8 if cfg.family == "DL" else 1.2, "ls": "-" if cfg.family == "DL" else "--"}
        name = cfg.label.split(" @")[0]
        fpr, tpr, _ = roc_curve(y, cfg.score)
        roc_ax.plot(fpr, tpr, label=f"{name}  AUC {point[cfg.key]['roc_auc']:.3f}", **style)
        prec, rec, _ = precision_recall_curve(y, cfg.score)
        pr_ax.plot(rec, prec, label=f"{name}  AP {point[cfg.key]['pr_auc']:.3f}", **style)
    for cfg, marker in marked:
        m, c = point[cfg.key], colour(cfg, dl_order)
        roc_ax.plot(1 - m["specificity"], m["sensitivity"], marker, color=c, ms=9, mec="black",
                    label=f"operating point {cfg.label}")
        pr_ax.plot(m["sensitivity"], m["precision"], marker, color=c, ms=9, mec="black")
    roc_ax.plot([0, 1], [0, 1], color="black", lw=0.6, ls=":")
    pr_ax.axhline(float(y.mean()), color="black", lw=0.6, ls=":")
    roc_ax.set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC, pooled LOSO test windows")
    pr_ax.set(xlabel="recall (sensitivity)", ylabel="precision", title="Precision-recall")
    roc_ax.legend(fontsize=7, loc="lower right")
    pr_ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_metric_bars(key_configs: list[Config], point: dict, samples: dict, path: Path,
                     dl_order: list[str]) -> None:
    plt = _pyplot()
    metrics = ("accuracy", "f1_macro", "f1_fall", "sensitivity", "specificity", "mcc", "roc_auc")
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x, width = np.arange(len(metrics)), 0.8 / len(key_configs)
    for i, cfg in enumerate(key_configs):
        vals = np.array([point[cfg.key][m] for m in metrics])
        lo = np.array([np.nanpercentile(samples[cfg.key][m], 2.5) for m in metrics])
        hi = np.array([np.nanpercentile(samples[cfg.key][m], 97.5) for m in metrics])
        err = np.clip(np.vstack([vals - lo, hi - vals]), 0, None)
        ax.bar(x + (i - (len(key_configs) - 1) / 2) * width, vals, width, yerr=err, capsize=2,
               color=colour(cfg, dl_order), edgecolor="black", lw=0.4, label=cfg.label,
               error_kw={"lw": 0.7})
    ax.set_xticks(x, [TITLES[m] for m in metrics])
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Pooled LOSO metrics; error bars = subject-level bootstrap 95 % interval")
    fig.legend(*ax.get_legend_handles_labels(), fontsize=8, ncol=len(key_configs),
               loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_per_subject(ps: pd.DataFrame, ml: Config, dl: Config, path: Path,
                     dl_order: list[str]) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ("roc_auc", "specificity"),
                                 ("ROC-AUC per subject (subjects with falls)",
                                  "Specificity per subject (val-tuned threshold)")):
        both = pd.concat([ps[ps["config"] == ml.key].set_index("subject")[metric],
                          ps[ps["config"] == dl.key].set_index("subject")[metric]],
                         axis=1, keys=["ml", "dl"]).dropna()
        ypos = np.arange(len(both))
        ax.hlines(ypos, both.min(axis=1), both.max(axis=1), color="0.75", lw=2)
        ax.plot(both["ml"], ypos, "o", color=colour(ml, dl_order), label=ml.label)
        ax.plot(both["dl"], ypos, "D", color=colour(dl, dl_order), label=dl.label)
        ax.set_yticks(ypos, [f"subject {int(s)}" for s in both.index])
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_confusions(key_configs: list[Config], point: dict, path: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(key_configs), figsize=(3.0 * len(key_configs), 3.1))
    for ax, cfg in zip(np.atleast_1d(axes), key_configs):
        m = point[cfg.key]
        grid = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        share = grid / grid.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v}\n{share[i, j]:.1%}", ha="center", va="center", fontsize=8,
                    color="white" if share[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1], ["pred. ADL", "pred. Fall"])
        ax.set_yticks([0, 1], ["ADL", "Fall"])
        ax.set_title(cfg.label, fontsize=8)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_activity(pa: pd.DataFrame, key_configs: list[Config], path: Path,
                  dl_order: list[str]) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 2.8]})
    panels = (("Fall", "impact_detection", "Falls: impact windows detected"),
              ("ADL", "positive_rate", "ADL: false-positive rate per window"))
    for ax, (cls, metric, title) in zip(axes, panels):
        frame = pa[pa["class"] == cls]
        acts = sorted(frame["activity"].unique())
        x, width = np.arange(len(acts)), 0.8 / len(key_configs)
        for i, cfg in enumerate(key_configs):
            vals = frame[frame["config"] == cfg.key].set_index("activity").loc[acts, metric]
            ax.bar(x + (i - (len(key_configs) - 1) / 2) * width, vals.to_numpy(), width,
                   color=colour(cfg, dl_order), edgecolor="black", lw=0.3, label=cfg.label)
        ax.set_xticks(x, acts, rotation=35, ha="right")
        ax.set_title(title)
    axes[0].set_ylim(0, 1.05)
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_events(points: dict, configs: list[Config], path: Path, dl_order: list[str]) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for cfg in configs:
        one, two = points[(cfg.key, 1)], points[(cfg.key, 2)]
        c = colour(cfg, dl_order)
        ax.plot([one["false_alarms_per_hour"], two["false_alarms_per_hour"]],
                [one["sensitivity"], two["sensitivity"]], color=c, lw=0.8,
                ls="-" if cfg.family == "DL" else "--")
        ax.plot(one["false_alarms_per_hour"], one["sensitivity"], "o", ms=7, mec="black", mew=0.4,
                color=c, label=cfg.label)
        ax.plot(two["false_alarms_per_hour"], two["sensitivity"], "s", ms=7, mec="black", mew=0.4,
                color=c)
    ax.set(xlabel=f"false alarms per hour of ADL ({COOLDOWN_WINDOWS} s cooldown)",
           ylabel="fall recordings with an alarm",
           title="Event level - circle: alarm on 1 positive window, square: on 2 in a row")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def f3(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return "-"
    return f"{v:.{digits}f}"

def md_table(headers: list[str], rows: list[list], align: str | None = None) -> list[str]:
    align = align or ("l" + "c" * (len(headers) - 1))
    sep = ["---" if a == "l" else ":---:" for a in align]
    return (["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
            + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])

def delta_cell(r, digits: int = 3) -> str:
    mark = {"DL higher": " ▲", "ML higher": " ▼", "DL fewer": " ▲", "ML fewer": " ▼"}.get(r["verdict"], "")

    def num(v: float) -> str:
        text = f"{v:+.{digits}f}"
        return f"{v:+.{digits + 2}f}" if v != 0 and float(text) == 0 else text

    return f"{num(r['delta'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]{mark}"

def pval(p: float) -> str:
    return "-" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")

def _unique_pairs(pairs) -> list[tuple[Config, Config]]:
    seen, out = set(), []
    for dl, ml in pairs:
        if ml is not None and (dl.key, ml.key) not in seen:
            seen.add((dl.key, ml.key))
            out.append((dl, ml))
    return out

def analyse_windows(R: SimpleNamespace) -> None:
    y, by_key = R.y, R.by_key
    R.impact, R.soft, R.cutoff = impact_masks(R.table)
    R.boot = Bootstrap(R.table["subject_id"].to_numpy())
    print(f"window level: {len(R.configs)} configurations, {BOOTSTRAP_ITERATIONS} bootstrap resamples",
          flush=True)
    R.point = {c.key: window_metrics(c, y, R.impact, R.soft) for c in R.configs}
    cache: dict = {}
    R.samples = {c.key: R.boot.window_samples(c, y, R.impact, R.soft, cache) for c in R.configs}

    R.deployed, R.rf_val = by_key["rf53@0.30"], by_key["rf53@val"]
    ml_configs = [c for c in R.configs if c.family == "ML"]
    R.pairs_main = [(R.main_dl, m) for m in ml_configs]
    R.pairs_matched = _unique_pairs((d, matched_forest(d, by_key)) for d in R.dl_configs)
    pairs = _unique_pairs(R.pairs_main + R.pairs_matched
                          + [(d, m) for d in R.dl_configs for m in (R.deployed, R.rf_val)])
    R.deltas = delta_table(R.point, R.samples, pairs)
    R.mcnemar = pd.DataFrame([mcnemar(a, b, y, R.boot.index, R.boot.n) for a, b in pairs])
    R.ps = per_subject(R.configs, y, R.boot)
    R.wilcoxon = wilcoxon_table(R.ps, _unique_pairs([(R.main_dl, R.rf_val), (R.main_dl, R.deployed)]
                                                    + R.pairs_matched))
    matched = matched_forest(R.main_dl, by_key)
    R.key_configs = [c for c in (R.deployed, R.rf_val, matched, by_key.get("svm_rbf@val"),
                                 R.ref_dl, R.main_dl) if c is not None]
    R.pa = per_activity(R.key_configs, R.table, R.impact)
    R.fixed = fixed_points(R.configs, y)
    R.fixed_pairs = _unique_pairs([(R.main_dl, R.rf_val), (R.main_dl, matched)] + R.pairs_matched)
    R.fixed_deltas = fixed_point_deltas(R, R.fixed_pairs)

def analyse_events(R: SimpleNamespace) -> None:
    print("event level: replaying every recording", flush=True)
    R.recs = build_recordings(R.table, R.boot)
    R.ev_points, R.ev_samples = {}, {}
    for c in R.configs:
        for confirm in (1, 2):
            R.ev_points[(c.key, confirm)], R.ev_samples[(c.key, confirm)] = event_metrics(
                c, R.recs, confirm, R.boot)
    R.adl_hours = R.ev_points[(R.configs[0].key, 1)]["adl_hours"]
    R.ev_pairs = _unique_pairs([(R.main_dl, R.deployed), (R.main_dl, R.rf_val)] + R.pairs_matched)
    R.ev_deltas = event_deltas(R.ev_points, R.ev_samples, R.ev_pairs)

    rec_rows = R.recs.rows_by_subject(R.boot.n)
    rows, auc_samples, seen = [], {}, set()
    for c in R.configs:
        if c.source in seen:
            continue
        seen.add(c.source)
        for confirm in (1, 2):
            s = recording_scores(c.score, R.recs, confirm)
            auc_samples[(c.source, confirm)] = R.boot.score_samples(R.recs.label, s, rec_rows)["roc_auc"]
            rows.append({"source": c.source, "model": c.label.split(" @")[0], "family": c.family,
                         "confirm": confirm, "roc_auc": auc(R.recs.label, s),
                         "sens_at_spec_0.90": _sens_at_spec(R.recs.label, s, 0.90),
                         "sens_at_spec_0.95": _sens_at_spec(R.recs.label, s, 0.95)})
    R.rec_auc = pd.DataFrame(rows)
    rows = []
    for dl, ml in _unique_pairs(R.ev_pairs):
        for confirm in (1, 2):
            d = auc_samples[(dl.source, confirm)] - auc_samples[(ml.source, confirm)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            get = R.rec_auc.set_index(["source", "confirm"])["roc_auc"]
            rows.append({"dl": dl.key, "dl_label": dl.label, "ml": ml.key, "ml_label": ml.label,
                         "confirm": confirm, "metric": "recording_roc_auc",
                         "delta": get[(dl.source, confirm)] - get[(ml.source, confirm)],
                         "ci_low": float(lo), "ci_high": float(hi), "verdict": verdict(lo, hi)})
    R.rec_auc_deltas = pd.DataFrame(rows)

def _seconds(s: float) -> str:
    return f"{s:.1f} s" if s < 120 else f"{s / 60:.1f} min"

def analyse_compute(R: SimpleNamespace, native: bool) -> None:
    print("compute: sizes and latency", flush=True)
    R.native = native_forest_latency() if native else None
    deployed = deployed_forest()
    card = json.loads(config.MODEL_CARD.read_text(encoding="utf-8")) if config.MODEL_CARD.is_file() else {}
    rows = []
    for spec in ML_SPECS:
        m, n_feat = R.ml_meta["models"][spec.key], len(feature_subset(spec.subset))
        row = {"model": spec.short, "family": "ML", "train": _seconds(m["fit_s"]),
               "inputs": f"{n_feat} features ({SUBSET_INPUTS[spec.subset]})",
               "latency": f"sklearn {m['infer_us']:.1f} us (batched)", "device": "not exported"}
        if spec.model == "random_forest":
            row.update(size=f"60 trees, {m['nodes']:,.0f} nodes", storage=f"{m['flash_kib']:.0f}",
                       work=f"~{m['node_tests']:.0f} comparisons + {n_feat} features")
            if spec.key == "rf53":
                row["device"] = "yes: [env:ml_inference]"
                if deployed["nodes"]:
                    row["size"] += f" (flashed: {deployed['nodes']:,})"
                    row["storage"] += f" (flashed: {deployed['flash_kib']:.0f})"
                if R.native:
                    row["latency"] = (f"C++ firmware code: {R.native['features_us']:.1f} us features "
                                      f"+ {R.native['forest_us']:.1f} us forest; {row['latency']}")
            elif spec.key == "rf38":
                row["device"] = "possible: export_model.py --axes acc_gyro"
        elif spec.model == "svm_linear":
            row.update(size=f"{n_feat + 1} weights (from {m['support_vectors']:,.0f} support vectors)",
                       storage=f"{(n_feat + 1) * 4 / 1024:.1f}", work=f"{n_feat} MACs")
        elif spec.model.startswith("svm"):
            sv = m["support_vectors"]
            row.update(size=f"{sv:,.0f} support vectors", storage=f"{sv * (n_feat + 1) * 4 / 1024:.0f}",
                       work=f"~{sv * n_feat / 1e3:.0f} k MACs + {sv:,.0f} exp")
        else:
            row.update(size=f"{n_feat + 1} weights", storage=f"{(n_feat + 1) * 4 / 1024:.1f}",
                       work=f"{n_feat} MACs")
        rows.append(row)
    for run in R.runs:
        c = dl_compute(run)
        row = {"model": run.short, "family": "DL", "size": f"{c['parameters']:,} parameters",
               "inputs": f"raw window {len(run.layout.raw_inputs)} x {config.WINDOW_SAMPLES} ({run.inputs})",
               "storage": f"{c['parameters'] * 4 / 1024:.0f}", "work": f"{c['macs'] / 1e6:.2f} M MACs",
               "latency": f"PyTorch {c['torch_ms_batch1']:.2f} ms (batch 1), {c['torch_us_batched']:.0f} us (batched)",
               "train": _seconds(c["train_s_per_fold"]), "device": "no: ONNX for a gateway / server"}
        if run.tag == R.shipped and card.get("onnx"):
            int8 = card.get("onnx_int8")
            row["storage"] += (f" (ONNX {card['onnx']['bytes'] / 1024:.0f}"
                               + (f", INT8 {int8['bytes'] / 1024:.0f}" if int8 else "") + ")")
            row["latency"] += f"; ONNX Runtime {card['parity']['latency_ms_batch1_1thread']:.2f} ms"
        rows.append(row)
    R.compute = pd.DataFrame(rows)

def _delta_row(frame: pd.DataFrame, dl: Config, ml: Config, metric: str, **extra):
    sel = (frame["dl"] == dl.key) & (frame["ml"] == ml.key) & (frame["metric"] == metric)
    for column, value in extra.items():
        sel &= frame[column] == value
    return frame[sel].iloc[0]

def pair_line(R: SimpleNamespace, dl: Config, ml: Config) -> str:
    sub = R.deltas[(R.deltas["dl"] == dl.key) & (R.deltas["ml"] == ml.key)
                   & R.deltas["metric"].isin(SUMMARY_METRICS)]
    better = [f"{TITLES[r.metric]} {r.delta:+.3f}" for r in sub.itertuples() if r.verdict == "DL higher"]
    worse = [f"{TITLES[r.metric]} {r.delta:+.3f}" for r in sub.itertuples() if r.verdict == "ML higher"]
    return (f"higher point estimate on {int((sub['delta'] > 0).sum())} of {len(sub)} metrics; "
            f"clearly higher (interval above 0): {', '.join(better) or 'none'}; "
            f"clearly lower (interval below 0): {', '.join(worse) or 'none'}.")

def section_summary(R: SimpleNamespace) -> list[str]:
    dl = R.main_dl
    rows = []
    for metric in SUMMARY_METRICS:
        best = {}
        for fam in ("ML", "DL"):
            fam_configs = [c for c in R.configs if c.family == fam]
            best[fam] = max(fam_configs, key=lambda c: np.nan_to_num(R.point[c.key][metric], nan=-1))
        rows.append([TITLES[metric]] + [f"{best[f].label}: {R.point[best[f].key][metric]:.3f}"
                                        for f in ("ML", "DL")])
    lines = ["## Summary", "", "Best configuration per metric (point estimates; whether a gap is "
             "real is decided by the intervals in section 3):", ""]
    lines += md_table(["Metric", "Best classic model", "Best deep model"], rows, "lll")
    lines += ["", f"**{dl.label}** ({dl.role or 'main deep model'}):", "",
              f"- vs **{R.deployed.label}**, the forest as deployed: " + pair_line(R, dl, R.deployed),
              f"- vs **{R.rf_val.label}**, same forest and same threshold rule: "
              + pair_line(R, dl, R.rf_val)]
    for d, m in R.pairs_matched:
        cells = [f"{TITLES[x]} {delta_cell(_delta_row(R.deltas, d, m, x))}" for x in ("roc_auc", "f1_macro")]
        lines.append(f"- same sensors ({d.inputs}), {d.label} vs {m.label}: " + "; ".join(cells))
    sub = R.deltas[(R.deltas["dl"] == dl.key) & R.deltas["ml"].isin([m.key for _, m in R.pairs_main])]
    tally = sub["verdict"].value_counts()
    lines.append(f"- over all {len(R.pairs_main)} classic configurations x {len(CI_METRICS)} metrics: "
                 f"{tally.get('DL higher', 0)} intervals favour the deep model, "
                 f"{tally.get('ML higher', 0)} the classic model, "
                 f"{tally.get('no clear difference', 0)} are inconclusive.")
    return lines

def section_summary_tail(R: SimpleNamespace) -> list[str]:
    dl = R.main_dl
    mc = R.mcnemar[(R.mcnemar["dl"] == dl.key) & (R.mcnemar["ml"] == R.rf_val.key)].iloc[0]
    lines = [f"- clustered McNemar test on window errors vs {R.rf_val.label}: p = "
             f"{pval(mc['p_clustered'])} ({mc['only_dl_correct']} windows only the deep model gets "
             f"right, {mc['only_ml_correct']} only the forest)."]
    fixed = R.fixed.set_index("source")
    for level in SENS_LEVELS[1:]:
        col = f"spec_at_sens_{level:.2f}"
        r = _delta_row(R.fixed_deltas, dl, R.rf_val, col)
        lines.append(f"- threshold-free, at a required window sensitivity of {level:.2f}: specificity "
                     f"{fixed.at[dl.source, col]:.3f} for {dl.label.split(' @')[0]} vs "
                     f"{fixed.at[R.rf_val.source, col]:.3f} for {R.rf_val.label.split(' @')[0]}, "
                     f"difference {delta_cell(r)}.")
    for confirm in (1, 2):
        a, b = R.ev_points[(dl.key, confirm)], R.ev_points[(R.deployed.key, confirm)]
        lines.append(f"- event level, alarm after {confirm} positive window(s) in a row: {dl.label} "
                     f"detects {a['sensitivity']:.3f} of fall trials with "
                     f"{a['false_alarms_per_hour']:.1f} false alarms per hour of ADL; "
                     f"{R.deployed.label}: {b['sensitivity']:.3f} with {b['false_alarms_per_hour']:.1f}/h.")
    rf = R.compute[R.compute["model"] == "RF-53"].iloc[0]
    deep = R.compute[R.compute["model"] == R.main_run.short].iloc[0]
    lines += [f"- cost per window: {deep['work']} for {dl.label} vs {rf['work']} for the forest; "
              "only the forest runs on the ESP32-C3 today.", ""]
    return lines

def section_protocol(R: SimpleNamespace) -> list[str]:
    sizes = R.table.groupby("subject_id").size()
    with_falls = set(R.table.loc[R.table["label"] == 1, "subject_id"].astype(int))
    never = sorted(with_falls - set(config.VALIDATION_POOL))
    lines = ["## 1. Protocol", "", "Identical for every model:", "",
             f"- **Data:** the {len(R.y)} windows of `ml_pipeline` ({int(R.y.sum())} Fall / "
             f"{int((R.y == 0).sum())} ADL) from {R.table['source_file'].nunique()} UMAFall wrist "
             f"recordings of {R.boot.n} subjects; 2.0 s windows, 1.0 s hop, file-level labels.",
             f"- **Folds:** leave-one-subject-out, {R.boot.n} folds. Per fold "
             f"{config.VALIDATION_SUBJECTS_PER_FOLD} validation subjects (drawn from the subjects with "
             f"falls{', never subject ' + ', '.join(map(str, never)) if never else ''}) and the rest "
             "for training. The test subject is never seen by a model or by threshold selection.",
             "- **Threshold:** Youden's J on the validation subjects (`@ val`); the deployed forest also "
             f"at its firmware threshold {config.ML_INFERENCE_THRESHOLD:.2f} (`@ 0.30`).",
             f"- **Pooling:** metrics are computed over all test windows at once (folds hold "
             f"{sizes.min()} to {sizes.max()} windows).",
             f"- **Uncertainty:** subject-level bootstrap, {BOOTSTRAP_ITERATIONS} resamples of the "
             f"{R.boot.n} subjects, paired across models; 95 % percentile intervals. ▲ = interval above "
             "zero (deep model higher), ▼ = below zero (classic model higher), no mark = inconclusive.",
             f"- **Check:** the forest refitted here reproduces the forest stored in every benchmark run "
             f"to {R.rf_diff:.1e} (probabilities and thresholds).", ""]
    rows = [[c.label, c.family, c.inputs, R.input_detail[c.key], c.role] for c in R.configs]
    lines += md_table(["Configuration", "Family", "Sensors", "Model input", "Role"], rows, "lllll")
    lines += ["", "SVM scores are the decision value passed through a logistic function: monotonic, so "
              "the ROC is unchanged and 0.5 is the SVM's own boundary, but not a probability, so no Brier "
              "score is given. RF-38 and RF-19 are the deployed forest's settings on the features of fewer "
              "sensors, so the deep models can be compared with a forest that sees the same sensors.", ""]
    return lines

def _bold_best(values: list[float], higher: bool = True) -> list[str]:
    finite = [v for v in values if np.isfinite(v)]
    best = (max(finite) if higher else min(finite)) if finite else None
    return [f"**{v:.3f}**" if best is not None and np.isfinite(v) and np.isclose(v, best) else f3(v)
            for v in values]

def section_window(R: SimpleNamespace) -> list[str]:
    lines = ["## 2. Window-level results", "",
             "Table 1 - thresholded metrics, pooled over all test windows (best per column in bold).", ""]
    cols = ("accuracy", "balanced_accuracy", "f1_macro", "f1_fall", "sensitivity", "specificity",
            "precision", "mcc")
    body = list(zip(*[_bold_best([R.point[c.key][m] for c in R.configs]) for m in cols]))
    lines += md_table(["Configuration"] + [TITLES[m] for m in cols],
                      [[c.label, *body[i]] for i, c in enumerate(R.configs)])
    lines += ["", "Table 2 - threshold-free metrics, sensitivity on the fall itself, and counts.", ""]
    cols2 = ("roc_auc", "pr_auc", "impact_sensitivity", "soft_fall_sensitivity")
    body = list(zip(*[_bold_best([R.point[c.key][m] for c in R.configs]) for m in cols2]))
    brier = _bold_best([R.point[c.key]["brier"] for c in R.configs], higher=False)
    rows = [[c.label, *body[i], brier[i]] + [R.point[c.key][k] for k in ("tp", "fp", "tn", "fn")]
            for i, c in enumerate(R.configs)]
    lines += md_table(["Configuration"] + [TITLES[m] for m in cols2] + ["Brier (lower = better)",
                                                                         "TP", "FP", "TN", "FN"], rows)
    lines += ["", f"Impact sens. = share of impact windows flagged (the windows that contain a fall's "
              f"peak |a|; {int(R.impact.sum())} windows). Soft-fall sens. = the same for the softest "
              f"third of falls (peak |a| <= {R.cutoff:.2f} g; {int(R.soft.sum())} windows). With "
              "file-level labels most Fall windows are the standing and lying around the impact; these "
              "two columns measure the fall itself.", "",
              "Table 3 - main configurations with 95 % intervals.", ""]
    cols3 = ("accuracy", "f1_macro", "f1_fall", "sensitivity", "specificity", "mcc", "roc_auc")
    rows = [[c.label] + [f"{R.point[c.key][m]:.3f} [{np.nanpercentile(R.samples[c.key][m], 2.5):.3f}, "
                         f"{np.nanpercentile(R.samples[c.key][m], 97.5):.3f}]" for m in cols3]
            for c in R.key_configs]
    lines += md_table(["Configuration"] + [TITLES[m] for m in cols3], rows)
    if R.figures:
        lines += ["", "![Main metrics with 95 % intervals](figures/metrics_ci.png)",
                  "", "![Confusion matrices](figures/confusion_matrices.png)"]
    return lines + [""]

def section_head_to_head(R: SimpleNamespace) -> list[str]:
    dl, mls = R.main_dl, [m for _, m in R.pairs_main]
    lines = ["## 3. Head-to-head: is the difference real?", "",
             f"Table 4 - **{dl.label}** minus each classic model; cell = difference [95 % interval].", ""]
    rows = [[TITLES[metric]] + [delta_cell(_delta_row(R.deltas, dl, m, metric)) for m in mls]
            for metric in CI_METRICS]
    lines += md_table(["Metric"] + [m.label for m in mls], rows)
    lines += ["", "Table 5 - same sensors on both sides: every deep model against the forest built from "
              "the same sensors' features (both @ val).", ""]
    cols = ("roc_auc", "f1_macro", "accuracy", "sensitivity", "specificity", "mcc")
    rows = [[d.label, m.label, d.inputs] + [delta_cell(_delta_row(R.deltas, d, m, x)) for x in cols]
            for d, m in R.pairs_matched]
    lines += md_table(["Deep model", "Forest", "Sensors"] + [TITLES[x] for x in cols], rows,
                      "lll" + "c" * len(cols))
    lines += ["", "Table 6 - every deep model against the deployed forest.", ""]
    cols = ("accuracy", "f1_macro", "f1_fall", "sensitivity", "specificity", "roc_auc")
    rows = [[d.label, m.label] + [delta_cell(_delta_row(R.deltas, d, m, x)) for x in cols]
            for d in R.dl_configs for m in (R.deployed, R.rf_val)]
    lines += md_table(["Deep model", "Forest"] + [TITLES[x] for x in cols], rows, "ll" + "c" * len(cols))
    lines += ["", "Table 7 - McNemar test on window-level errors (each model at its own threshold).", ""]
    wanted = {(dl.key, m.key) for m in mls} | {(d.key, m.key) for d, m in R.pairs_matched}
    rows = [[r.dl_label, r.ml_label, r.only_dl_correct, r.only_ml_correct, pval(r.p_mcnemar),
             pval(r.p_clustered)] for r in R.mcnemar.itertuples() if (r.dl, r.ml) in wanted]
    lines += md_table(["Deep model", "Classic model", "Only DL correct", "Only ML correct",
                       "p, windows independent", "p, clustered by subject"], rows, "llcccc")
    lines += ["", "The first p-value treats the overlapping windows as independent and is too small; "
              "the clustered test (Durkalski et al., 2003) counts each subject as one cluster and is "
              "the one to read.", ""]
    return lines

def section_threshold_free(R: SimpleNamespace) -> list[str]:
    lines = ["## 4. Threshold-free comparison", "",
             "Table 8 - each model's pooled ROC curve read at fixed operating points (best per column "
             "in bold). This compares the curves themselves, wherever a threshold was put; the levels "
             "are read off the pooled test curves, so they are not deployable thresholds.", ""]
    cols = [f"spec_at_sens_{v:.2f}" for v in SENS_LEVELS] + [f"sens_at_spec_{v:.2f}" for v in SPEC_LEVELS]
    heads = [f"Spec. @ sens. {v:.2f}" for v in SENS_LEVELS] + [f"Sens. @ spec. {v:.2f}" for v in SPEC_LEVELS]
    body = list(zip(*[_bold_best(R.fixed[c].tolist()) for c in cols]))
    rows = [[r["model"], r["family"], *body[i]] for i, (_, r) in enumerate(R.fixed.iterrows())]
    lines += md_table(["Model", "Family"] + heads, rows, "ll" + "c" * len(cols))
    lines += ["", "Table 9 - the same read-outs as differences, DL - ML [95 % interval]. The forest's "
              "curve does not depend on its threshold, so RF-53 @ 0.30 and RF-53 @ val share a row.", ""]
    rows = [[dl.label.split(" @")[0], ml.label.split(" @")[0]]
            + [delta_cell(_delta_row(R.fixed_deltas, dl, ml, c)) for c in cols]
            for dl, ml in R.fixed_pairs]
    lines += md_table(["Deep model", "Classic model"] + heads, rows, "ll" + "c" * len(cols))
    if R.figures:
        lines += ["", "![ROC and precision-recall curves](figures/roc_pr.png)"]
    return lines + [""]

def section_subjects(R: SimpleNamespace) -> list[str]:
    dl, ml = R.main_dl, R.rf_val
    a = R.ps[R.ps["config"] == ml.key].set_index("subject")
    b = R.ps[R.ps["config"] == dl.key].set_index("subject")
    lines = ["## 5. Per subject", "", f"Table 10 - {ml.label} vs {dl.label} on each held-out subject.", ""]
    rows = [[s, int(a.at[s, "windows"]), int(a.at[s, "falls"])]
            + [f3(frame.at[s, m]) for m in ("roc_auc", "sensitivity", "specificity") for frame in (a, b)]
            for s in a.index]
    lines += md_table(["Subject", "Windows", "Fall windows", "AUC RF", "AUC DL", "Sens. RF", "Sens. DL",
                       "Spec. RF", "Spec. DL"], rows)
    lines += ["", "Table 11 - Wilcoxon signed-rank test on per-subject differences (two-sided).", ""]
    rows = [[r.dl_label, r.ml_label, TITLES[r.metric], r.subjects, r.dl_better, r.ml_better,
             f"{r.median_delta:+.3f}", pval(r.p)] for r in R.wilcoxon.itertuples()]
    lines += md_table(["Deep model", "Classic model", "Metric", "Subjects", "DL better", "ML better",
                       "Median difference", "p"], rows, "lll" + "c" * 5)
    no_falls = [int(s) for s in a.index if a.at[s, "falls"] == 0]
    lines += ["", f"Subjects {', '.join(map(str, no_falls))} recorded no falls: they have no sensitivity "
              "or AUC, so those rows rest on the remaining subjects. With so few subjects a Wilcoxon "
              "test can only detect large, consistent differences."]
    if R.figures:
        lines += ["", "![Per-subject AUC and specificity](figures/per_subject.png)"]
    return lines + [""]

def section_activity(R: SimpleNamespace) -> list[str]:
    heads = [c.label for c in R.key_configs]
    lines = ["## 6. Per activity", "", "Table 12 - falls: impact windows detected / all fall windows "
             "flagged.", ""]
    rows = []
    for act, g in R.pa[R.pa["class"] == "Fall"].groupby("activity"):
        g = g.set_index("config")
        rows.append([act, int(g["recordings"].iloc[0])]
                    + [f"{g.at[c.key, 'impact_detection']:.2f} / {g.at[c.key, 'positive_rate']:.2f}"
                       for c in R.key_configs])
    lines += md_table(["Fall type", "Recordings"] + heads, rows)
    lines += ["", "Table 13 - ADL: false-positive rate per window, lower is better (best in bold); "
              "sorted by the mean over the columns.", ""]
    adl = R.pa[R.pa["class"] == "ADL"]
    order = adl.groupby("activity")["positive_rate"].mean().sort_values(ascending=False).index
    rows = []
    for act in order:
        g = adl[adl["activity"] == act].set_index("config")
        rows.append([act, int(g["recordings"].iloc[0])]
                    + _bold_best([g.at[c.key, "positive_rate"] for c in R.key_configs], higher=False))
    lines += md_table(["Activity", "Recordings"] + heads, rows)
    if R.figures:
        lines += ["", "![Per activity](figures/per_activity.png)"]
    return lines + [""]

def section_events(R: SimpleNamespace) -> list[str]:
    n_fall, n_adl = int((R.recs.label == 1).sum()), int((R.recs.label == 0).sum())
    lines = ["## 7. Event level: one alarm decision per recording", "",
             "The device raises alarms, not window labels. Each recording (a ~14 s trial) is replayed "
             "window by window: an alarm needs 1 positive window (`CONFIRMATION_WINDOWS = 1`, as in "
             "`inference_main.cpp`) or 2 in a row, and is followed by a "
             f"{COOLDOWN_WINDOWS} s cooldown. A fall trial is detected if it raised at least one alarm; "
             "an ADL trial is a false alarm if it raised any. False alarms per hour count every alarm "
             f"during the {R.adl_hours:.2f} h of ADL recordings.", "",
             "Table 14 - recording level, at each model's window-level threshold.", ""]
    rows = []
    for confirm in (1, 2):
        for c in R.configs:
            m = R.ev_points[(c.key, confirm)]
            rows.append([c.label, confirm, f3(m["sensitivity"]), f3(m["specificity"]), m["fn"], m["fp"],
                         f"{m['false_alarms_per_hour']:.1f}"])
    ref = RULE_BASED_REFERENCE
    rows.append(["Rule-based cascade, raw_stream (README 9.4)", "-", f3(ref["sensitivity"]),
                 f3(ref["specificity"]), ref["fn"], ref["fp"], "not measured"])
    lines += md_table(["Configuration", "Windows in a row", "Fall trials detected",
                       "ADL trials without alarm", f"Missed falls (of {n_fall})",
                       f"ADL trials with alarm (of {n_adl})", "False alarms / h"], rows, "lcccccc")
    lines += ["", "The window thresholds were tuned for windows, not for trials, so these are not each "
              "model's best event-level setting. The rule-based row is quoted from README 9.4 (separate "
              "run, same recordings and question, its own thresholds chosen by LOSO), for reference.", "",
              "Table 15 - recording-level ROC-AUC, threshold-free: the recording score is its highest "
              "window probability (1 window) or its best pair of consecutive windows (2 in a row).", ""]
    wide = R.rec_auc.pivot_table(index=["model", "family"], columns="confirm",
                                 values=["roc_auc", "sens_at_spec_0.95"], sort=False)
    rows = [[model, fam] + [f3(wide.loc[(model, fam), (v, k)]) for v in ("roc_auc", "sens_at_spec_0.95")
                            for k in (1, 2)] for model, fam in wide.index]
    lines += md_table(["Model", "Family", "AUC, 1 window", "AUC, 2 in a row", "Sens. @ spec. 0.95, 1 window",
                       "Sens. @ spec. 0.95, 2 in a row"], rows, "llcccc")
    lines += ["", "Table 16 - event-level differences, DL - ML [95 % interval]. For false alarms per hour "
              "lower is better: ▲ there means the deep model raises fewer.", ""]
    rows = []
    for dl, ml in R.ev_pairs:
        for confirm in (1, 2):
            cells = [delta_cell(_delta_row(R.ev_deltas, dl, ml, x, confirm=confirm),
                                1 if x == "false_alarms_per_hour" else 3) for x in EVENT_METRICS]
            auc_row = _delta_row(R.rec_auc_deltas, dl, ml, "recording_roc_auc", confirm=confirm)
            rows.append([dl.label, ml.label, confirm, *cells, delta_cell(auc_row)])
    lines += md_table(["Deep model", "Classic model", "In a row"] + [TITLES[x] for x in EVENT_METRICS]
                      + ["Recording AUC"], rows, "llc" + "c" * (len(EVENT_METRICS) + 1))
    if R.figures:
        lines += ["", "![Event-level trade-off](figures/event_level.png)"]
    return lines + [""]

def section_compute(R: SimpleNamespace) -> list[str]:
    lines = ["## 8. Computational cost and deployability", "",
             "Table 17 - model size, work and latency per 2 s window (this PC, one thread), training "
             "time per fold.", ""]
    rows = [[r.model, r.inputs, r.size, r.storage, r.work, r.latency, r.train, r.device]
            for r in R.compute.itertuples()]
    lines += md_table(["Model", "Input", "Size", "Storage (KiB)", "Work per window",
                       "Latency per window", "Training per fold", "On the ESP32-C3"], rows, "l" * 8)
    native = (f"compiled with {R.native['compiler']}; the forest's own self-test "
              f"{'passed' if R.native['selftest'] else 'FAILED'}" if R.native else "not run")
    lines += ["", "- *Forest sizes* are the median over the 19 fold models (trained on 15 subjects); "
              "\"flashed\" is the header in `include/RandomForest.h`, trained on all subjects. Deep-model "
              "storage is 4 bytes per float32 parameter.",
              f"- *C++ firmware code* = `feature_extractor.cpp` + `RandomForest.h`, the code "
              f"`[env:ml_inference]` runs, timed on this PC ({native}).",
              "- *PyTorch* latencies are measured on freshly built networks of each architecture; the "
              "cost does not depend on the weights. *ONNX Runtime* is the exported model "
              "(`model_card.json`).",
              f"- Training times were measured with folds running in parallel ({R.ml_meta['workers']} "
              "worker processes for the classic models, 4-5 processes x 2 threads for the deep ones), "
              "so they are comparable only in order of magnitude.",
              "- None of these timings is from the ESP32-C3 (160 MHz RISC-V, no FPU). The deep models "
              "have not been ported; 1-2 M multiply-accumulates in emulated float per second of data "
              "would need int8 quantisation and an on-target measurement first.", ""]
    return lines

def section_limitations() -> list[str]:
    return ["## 9. Limitations", "",
            "- **Dataset, not device.** UMAFall wrist data (~20 Hz, 19 subjects, scripted trials); nothing "
            "here is measured on Mitifall recordings.",
            "- **File-level labels.** Every window of a fall trial counts as Fall, also the standing and "
            "lying around the impact; the impact and soft-fall columns and the event level measure the "
            "fall itself.",
            "- **19 subjects.** Intervals are several points wide, and differences of 1-2 points between "
            "configurations are within them.",
            "- **Selection.** The shipped deep model was chosen after comparing four deep "
            "configurations on these folds; the classic baselines were not tuned here. Its margins are "
            "therefore slightly optimistic. The pre-specified TCN is the unselected reference.",
            "- **Pooled curves.** ROC/PR-AUC and the fixed operating points pool scores from 19 "
            "differently trained models, as is usual for LOSO; per-subject AUCs are in section 5.",
            "- **Event level.** ADL trials are ~14 s scripted activities, so false alarms per hour "
            "extrapolate from 2 h of data and say little about a day of real wear.", ""]

def write_csvs(R: SimpleNamespace) -> None:
    rows = []
    for c in R.configs:
        row = {"config": c.key, "label": c.label, "family": c.family, "inputs": c.inputs,
               "role": c.role, **R.point[c.key]}
        for m in CI_METRICS:
            row[f"{m}_ci_low"] = float(np.nanpercentile(R.samples[c.key][m], 2.5))
            row[f"{m}_ci_high"] = float(np.nanpercentile(R.samples[c.key][m], 97.5))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "window_metrics.csv", index=False)
    for name, frame in (("deltas", R.deltas), ("mcnemar", R.mcnemar), ("per_subject", R.ps),
                        ("wilcoxon", R.wilcoxon), ("per_activity", R.pa),
                        ("fixed_operating_points", R.fixed),
                        ("fixed_operating_point_deltas", R.fixed_deltas), ("recording_auc", R.rec_auc),
                        ("compute", R.compute)):
        frame.to_csv(OUT_DIR / f"{name}.csv", index=False)
    pd.DataFrame([{"config": k, "confirm": n, **v} for (k, n), v in R.ev_points.items()]).to_csv(
        OUT_DIR / "event_level.csv", index=False)
    pd.concat([R.ev_deltas, R.rec_auc_deltas], ignore_index=True).to_csv(
        OUT_DIR / "event_deltas.csv", index=False)

def write_figures(R: SimpleNamespace) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = [c.source for c in R.dl_configs]
    plot_curves(R.configs, R.y, R.point, [(R.deployed, "X"), (R.rf_val, "o"), (R.main_dl, "D")],
                FIG_DIR / "roc_pr.png", order)
    plot_metric_bars(R.key_configs, R.point, R.samples, FIG_DIR / "metrics_ci.png", order)
    plot_per_subject(R.ps, R.rf_val, R.main_dl, FIG_DIR / "per_subject.png", order)
    plot_confusions(R.key_configs, R.point, FIG_DIR / "confusion_matrices.png")
    activity_configs = [c for c in (R.deployed, R.rf_val, matched_forest(R.main_dl, R.by_key),
                                    R.main_dl) if c is not None]
    plot_activity(R.pa, activity_configs, FIG_DIR / "per_activity.png", order)
    plot_events(R.ev_points, R.configs, FIG_DIR / "event_level.png", order)

def write_report(R: SimpleNamespace) -> Path:
    lines = ["# Evaluation: deep learning vs classic machine learning", "",
             f"_Generated {R.created} by `python deep_learning/evaluation_report.py`. Every number is "
             "computed from saved predictions: the deep models' from `outputs/benchmark/*/predictions.csv`, "
             "the classic models' refitted on the same folds (`outputs/evaluation/ml_predictions.csv`). "
             "Each table is also written as a CSV next to this file._", ""]
    lines += section_summary(R) + section_summary_tail(R) + section_protocol(R) + section_window(R)
    lines += section_head_to_head(R) + section_threshold_free(R) + section_subjects(R)
    lines += section_activity(R) + section_events(R) + section_compute(R) + section_limitations()
    path = OUT_DIR / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluation report: deep_learning vs the classic ML models.")
    p.add_argument("--refit-ml", action="store_true", help="refit the classic models even when cached")
    p.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)),
                   help="worker processes for refitting the classic models")
    p.add_argument("--shipped", default=None,
                   help="benchmark tag of the main deep model (default: the one in model_card.json)")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-native", action="store_true", help="skip timing the firmware forest with g++")
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    table = load_ml_feature_table()
    R = SimpleNamespace(table=table, y=table["label"].to_numpy(int), figures=not args.no_figures,
                        created=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    R.runs = load_dl_runs(table)
    R.ml, R.ml_meta = classic_predictions(table, args.refit_ml, args.jobs)
    R.rf_diff = rf_consistency(R.ml, R.runs)
    if R.rf_diff > 1e-9:
        print(f"WARNING: the refitted forest differs from the benchmark copy by {R.rf_diff:.2e}")
    R.shipped = args.shipped or shipped_tag()
    prespecified = run_tag(benchmark_args([]))
    R.configs = build_configs(R.ml, R.runs, R.shipped, prespecified)
    R.by_key = {c.key: c for c in R.configs}
    R.dl_configs = [c for c in R.configs if c.family == "DL"]
    R.main_dl = R.by_key.get(R.shipped) or R.by_key.get(prespecified) or R.dl_configs[0]
    R.main_run = next(r for r in R.runs if r.tag == R.main_dl.key)
    R.ref_dl = R.by_key.get(prespecified) if prespecified != R.main_dl.key else None
    R.input_detail = {}
    for c in R.configs:
        if c.family == "ML":
            spec = next(s for s in ML_SPECS if s.key == c.source)
            R.input_detail[c.key] = f"{len(feature_subset(spec.subset))} features"
        else:
            layout = next(r for r in R.runs if r.tag == c.key).layout
            R.input_detail[c.key] = (f"raw {len(layout.raw_inputs)} x {config.WINDOW_SAMPLES} -> "
                                     f"{len(layout.model_channels)} channels")

    analyse_windows(R)
    analyse_events(R)
    analyse_compute(R, native=not args.no_native)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csvs(R)
    if R.figures:
        write_figures(R)
    path = write_report(R)
    print(f"report: {path} ({time.perf_counter() - started:.0f} s)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
