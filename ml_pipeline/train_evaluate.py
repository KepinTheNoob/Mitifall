"""Model benchmarking under Leave-One-Subject-Out cross-validation.

Why LOSO: windows from a single recording are highly correlated, and windows
from the same subject share gait/posture signatures. A random split would leak
subject identity into the test set and inflate every metric. ``LeaveOneGroupOut``
grouped on ``subject_id`` gives one fold per subject, so every score is measured
on a person the model has never seen.

Metric semantics for fall detection:
  * sensitivity / recall = TP / (TP + FN)  -> falls actually caught
  * specificity          = TN / (TN + FP)  -> false-alarm behaviour
Both are reported per fold, pooled over all folds, and averaged across folds.

Six of the 19 UMAFall subjects contributed no Fall recordings, so their folds
contain no positives and sensitivity is mathematically undefined there. Those
folds report NaN instead of a misleading 0.0, fold averages are NaN-aware, and
the pooled confusion matrix (which is immune to the issue) is treated as the
headline number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from . import config
    from .feature_extraction import FEATURE_NAMES, load_feature_table
except ImportError:  # pragma: no cover - direct script execution
    import config
    from feature_extraction import FEATURE_NAMES, load_feature_table

METRIC_COLUMNS = (
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "balanced_accuracy",
)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def build_models(
    names: tuple[str, ...] = config.MODEL_NAMES, n_jobs: int = -1
) -> dict[str, Pipeline]:
    """Instantiate the benchmark models.

    Trees are scale-invariant, so the forest skips standardisation; the SVMs and
    the logistic model are fitted inside a ``Pipeline`` with ``StandardScaler``
    so the scaler is re-fitted on training folds only (no leakage).
    """
    factories: dict[str, callable] = {
        "random_forest": lambda: Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        random_state=config.RANDOM_STATE, n_jobs=n_jobs, **config.RF_PARAMS
                    ),
                )
            ]
        ),
        "svm_rbf": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(random_state=config.RANDOM_STATE, **config.SVM_RBF_PARAMS)),
            ]
        ),
        "svm_linear": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(random_state=config.RANDOM_STATE, **config.SVM_LINEAR_PARAMS)),
            ]
        ),
        "logistic_regression": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        random_state=config.RANDOM_STATE, **config.LOGREG_PARAMS
                    ),
                ),
            ]
        ),
    }

    unknown = [n for n in names if n not in factories]
    if unknown:
        raise ValueError(f"unknown model(s): {unknown}. Available: {list(factories)}")

    return {name: factories[name]() for name in names}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """2x2 counts, explicit about labels so empty classes stay representable."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else float("nan")


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Derive fall-detection metrics; undefined quantities become NaN.

    When a fold contains no positive ground truth (a subject with no Fall
    recordings), every positive-class metric is reported as NaN rather than 0.0.
    Precision would otherwise evaluate to ``0 / fp = 0.0`` and silently drag the
    fold-averaged precision down even though the model behaved correctly.
    """
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    total = tn + fp + fn + tp
    nan = float("nan")

    has_positives = (tp + fn) > 0
    sensitivity = _ratio(tp, tp + fn) if has_positives else nan
    specificity = _ratio(tn, tn + fp)
    precision = (_ratio(tp, tp + fp) if (tp + fp) else nan) if has_positives else nan

    if np.isnan(sensitivity) or np.isnan(precision):
        f1 = nan
    elif precision + sensitivity == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)

    balanced_accuracy = (
        nan
        if np.isnan(sensitivity) or np.isnan(specificity)
        else 0.5 * (sensitivity + specificity)
    )

    return {
        "accuracy": _ratio(tp + tn, total),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


# --------------------------------------------------------------------------- #
# LOSO cross-validation
# --------------------------------------------------------------------------- #


def run_loso(
    table: pd.DataFrame,
    models: dict[str, Pipeline],
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Run LOSO-CV for every model.

    Returns the per-fold metric table and the pooled confusion matrix per model.
    """
    X = table[FEATURE_NAMES].to_numpy(dtype=float)
    y = table["label"].to_numpy(dtype=int)
    groups = table["subject_id"].to_numpy()

    splitter = LeaveOneGroupOut()
    folds = list(splitter.split(X, y, groups))
    n_folds = len(folds)

    if verbose:
        print(
            f"LOSO-CV: {n_folds} folds (one per subject), "
            f"{len(table)} windows, {len(FEATURE_NAMES)} features"
        )

    fold_rows: list[dict[str, object]] = []
    pooled: dict[str, dict[str, int]] = {
        name: {"tn": 0, "fp": 0, "fn": 0, "tp": 0} for name in models
    }

    for model_name, prototype in models.items():
        started = time.perf_counter()
        for train_idx, test_idx in folds:
            subject = groups[test_idx][0]
            model = clone_pipeline(prototype)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])

            counts = confusion_counts(y[test_idx], y_pred)
            metrics = metrics_from_counts(counts)
            for key in pooled[model_name]:
                pooled[model_name][key] += counts[key]

            fold_rows.append(
                {
                    "model": model_name,
                    "test_subject": int(subject),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_test_fall": int((y[test_idx] == 1).sum()),
                    "n_test_adl": int((y[test_idx] == 0).sum()),
                    **counts,
                    **metrics,
                }
            )

        if verbose:
            elapsed = time.perf_counter() - started
            pooled_metrics = metrics_from_counts(pooled[model_name])
            print(
                f"  {model_name:<20s} {elapsed:6.1f}s  "
                f"pooled sens={pooled_metrics['sensitivity']:.3f} "
                f"spec={pooled_metrics['specificity']:.3f} "
                f"f1={pooled_metrics['f1']:.3f}"
            )

    return pd.DataFrame(fold_rows), pooled


def clone_pipeline(pipeline: Pipeline) -> Pipeline:
    """Fresh unfitted copy, so folds never share fitted state."""
    from sklearn.base import clone

    return clone(pipeline)


# --------------------------------------------------------------------------- #
# Aggregation / reporting
# --------------------------------------------------------------------------- #


def summarize(
    fold_metrics: pd.DataFrame, pooled: dict[str, dict[str, int]]
) -> pd.DataFrame:
    """Build the cross-model summary table.

    ``pooled_*`` columns come from the summed confusion matrix over all folds
    (every window counted exactly once). ``mean_*`` / ``std_*`` columns are the
    NaN-aware fold averages, which show per-subject variability.
    """
    rows: list[dict[str, object]] = []
    for model_name, counts in pooled.items():
        folds = fold_metrics[fold_metrics["model"] == model_name]
        row: dict[str, object] = {"model": model_name}
        row.update({f"pooled_{k}": v for k, v in metrics_from_counts(counts).items()})
        row.update({f"cm_{k}": v for k, v in counts.items()})
        for metric in METRIC_COLUMNS:
            values = folds[metric].to_numpy(dtype=float)
            defined = values[~np.isnan(values)]
            row[f"mean_{metric}"] = float(defined.mean()) if defined.size else float("nan")
            row[f"std_{metric}"] = float(defined.std()) if defined.size else float("nan")
            row[f"n_folds_{metric}"] = int(defined.size)
        row["folds"] = int(len(folds))
        row["folds_with_falls"] = int((folds["n_test_fall"] > 0).sum())
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values("pooled_f1", ascending=False).reset_index(drop=True)


def format_confusion_matrix(model_name: str, counts: dict[str, int]) -> str:
    """Human-readable aggregated confusion matrix."""
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    return "\n".join(
        [
            f"{model_name} (aggregated over all LOSO folds)",
            "                 pred ADL   pred Fall",
            f"  true ADL     {tn:10d} {fp:11d}",
            f"  true Fall    {fn:10d} {tp:11d}",
            f"  missed falls (FN): {fn}    false alarms (FP): {fp}",
        ]
    )


def performance_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compact table for console/report display."""
    view = summary[
        [
            "model",
            "pooled_sensitivity",
            "pooled_specificity",
            "pooled_f1",
            "pooled_accuracy",
            "pooled_precision",
            "pooled_balanced_accuracy",
            "cm_fn",
            "cm_fp",
        ]
    ].copy()
    view.columns = [
        "model",
        "sensitivity",
        "specificity",
        "f1",
        "accuracy",
        "precision",
        "bal_acc",
        "missed_falls",
        "false_alarms",
    ]
    for column in ("sensitivity", "specificity", "f1", "accuracy", "precision", "bal_acc"):
        view[column] = view[column].astype(float).round(4)
    return view


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table (avoids the optional tabulate dep)."""
    columns = [str(c) for c in frame.columns]
    cells = [[_format_cell(v) for v in row] for row in frame.itertuples(index=False)]
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in cells)) if cells else len(columns[i])
        for i in range(len(columns))
    ]

    def line(values: list[str]) -> str:
        padded = [value.ljust(widths[i]) for i, value in enumerate(values)]
        return "| " + " | ".join(padded) + " |"

    out = [line(columns), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(row) for row in cells]
    return "\n".join(out)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return "n/a" if np.isnan(value) else f"{value:.4f}"
    return str(value)


def write_report(
    report_path: Path,
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    pooled: dict[str, dict[str, int]],
    table: pd.DataFrame,
) -> None:
    """Markdown summary of the benchmark run."""
    subjects = sorted(int(s) for s in table["subject_id"].unique())
    lines = [
        "# LOSO-CV fall detection benchmark",
        "",
        f"- Windows: {len(table)} "
        f"({int((table.label == 1).sum())} Fall / {int((table.label == 0).sum())} ADL)",
        f"- Features: {len(FEATURE_NAMES)}",
        f"- Subjects (folds): {len(subjects)} -> {subjects}",
        "",
        "## Pooled performance (all folds combined)",
        "",
        markdown_table(performance_table(summary)),
        "",
        "## Fold-averaged performance (mean +/- std over subjects, NaN-aware)",
        "",
    ]

    averaged = summary[["model", *[f"mean_{m}" for m in METRIC_COLUMNS]]].copy()
    stds = summary[[f"std_{m}" for m in METRIC_COLUMNS]]
    for metric in METRIC_COLUMNS:
        averaged[metric] = [
            f"{m:.3f} +/- {s:.3f}"
            for m, s in zip(averaged[f"mean_{metric}"], stds[f"std_{metric}"])
        ]
        averaged = averaged.drop(columns=[f"mean_{metric}"])
    lines += [markdown_table(averaged), "", "## Aggregated confusion matrices", ""]

    for model_name in summary["model"]:
        lines += ["```", format_confusion_matrix(model_name, pooled[model_name]), "```", ""]

    best_model = str(summary.iloc[0]["model"])
    best_folds = fold_metrics[fold_metrics["model"] == best_model][
        [
            "test_subject",
            "n_test",
            "n_test_fall",
            "sensitivity",
            "specificity",
            "f1",
            "accuracy",
            "fn",
            "fp",
        ]
    ].sort_values("test_subject")
    lines += [
        f"## Per-subject folds for the best model ({best_model})",
        "",
        markdown_table(best_folds),
        "",
    ]

    folds_without_falls = sorted(
        int(s) for s in fold_metrics.loc[fold_metrics["n_test_fall"] == 0, "test_subject"].unique()
    )
    if folds_without_falls:
        lines += [
            "## Note on undefined folds",
            "",
            "Subjects with no Fall recordings produce folds without positive "
            "samples, so sensitivity/precision/F1 are undefined (NaN) there and "
            "are excluded from the fold averages: "
            f"{folds_without_falls}.",
            "",
        ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_results(
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    pooled: dict[str, dict[str, int]],
    table: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Persist fold metrics, the summary table, confusion matrices and report."""
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "fold_metrics": output_dir / config.FOLD_METRICS_CSV.name,
        "summary": output_dir / config.SUMMARY_CSV.name,
        "confusion": output_dir / config.CONFUSION_JSON.name,
        "report": output_dir / config.REPORT_MD.name,
    }

    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["confusion"].write_text(
        json.dumps(
            {
                name: {
                    "counts": counts,
                    "matrix": [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]],
                    "metrics": metrics_from_counts(counts),
                }
                for name, counts in pooled.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(paths["report"], summary, fold_metrics, pooled, table)
    return paths


def print_results(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    pooled: dict[str, dict[str, int]],
) -> None:
    print("\n" + "=" * 78)
    print("Pooled performance across all LOSO folds")
    print("=" * 78)
    print(performance_table(summary).to_string(index=False))

    print("\nFold-averaged (mean +/- std over subjects, NaN-aware)")
    for _, row in summary.iterrows():
        print(
            f"  {row['model']:<20s} "
            f"sens={row['mean_sensitivity']:.3f}+/-{row['std_sensitivity']:.3f}  "
            f"spec={row['mean_specificity']:.3f}+/-{row['std_specificity']:.3f}  "
            f"f1={row['mean_f1']:.3f}+/-{row['std_f1']:.3f}  "
            f"acc={row['mean_accuracy']:.3f}+/-{row['std_accuracy']:.3f}"
            f"   [{int(row['folds_with_falls'])}/{int(row['folds'])} folds contain falls]"
        )

    print("\nAggregated confusion matrices")
    for model_name in summary["model"]:
        print()
        print(format_confusion_matrix(model_name, pooled[model_name]))

    best = summary.iloc[0]
    print(
        f"\nBest pooled F1: {best['model']} "
        f"(F1={best['pooled_f1']:.4f}, sensitivity={best['pooled_sensitivity']:.4f}, "
        f"specificity={best['pooled_specificity']:.4f})"
    )

    scored = fold_metrics.dropna(subset=["f1"])
    worst = (
        fold_metrics.loc[scored.groupby("model")["f1"].idxmin()]
        if not scored.empty
        else scored
    )
    if not worst.empty:
        print("\nWorst fold per model (lowest F1):")
        for _, row in worst.iterrows():
            print(
                f"  {row['model']:<20s} subject {int(row['test_subject']):02d}: "
                f"f1={row['f1']:.3f} sens={row['sensitivity']:.3f} spec={row['specificity']:.3f}"
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark lightweight classifiers with Leave-One-Subject-Out CV."
    )
    parser.add_argument("--features-csv", type=Path, default=config.FEATURES_CSV)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(config.MODEL_NAMES),
        choices=list(config.MODEL_NAMES),
        help="subset of models to benchmark (default: all)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="parallel jobs for the random forest (default: %(default)s)",
    )
    return parser.parse_args(argv)


def evaluate(
    table: pd.DataFrame,
    model_names: tuple[str, ...] = config.MODEL_NAMES,
    output_dir: Path | None = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    """Full evaluation entry point reused by run_pipeline.py."""
    missing = [c for c in (*FEATURE_NAMES, "label", "subject_id") if c not in table.columns]
    if missing:
        raise ValueError(f"feature table is missing columns: {missing[:5]}")

    models = build_models(tuple(model_names), n_jobs=n_jobs)
    fold_metrics, pooled = run_loso(table, models, verbose=verbose)
    summary = summarize(fold_metrics, pooled)

    paths = save_results(fold_metrics, summary, pooled, table, output_dir)
    if verbose:
        print_results(summary, fold_metrics, pooled)
        print("\nArtifacts:")
        for label, path in paths.items():
            print(f"  {label:<13s} {path}")

    return summary, fold_metrics, pooled


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    table = load_feature_table(args.features_csv)
    evaluate(
        table,
        model_names=tuple(args.models),
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
