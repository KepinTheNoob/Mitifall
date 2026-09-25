"""Export the trained Random Forest to a self-contained C++ header for the ESP32-C3.

What this produces
------------------
* ``include/RandomForest.h``       - flat-array decision forest + probability API
* ``include/features_order.json``  - the exact feature sequence ``predict()`` expects

Why a hand-rolled exporter instead of micromlgen / m2cgen
---------------------------------------------------------
``micromlgen``'s random-forest port emits nested ``if``/``else`` code that only
returns an ``argmax`` class label, so there is no probability to threshold on -
which is exactly what this project needs. ``m2cgen`` does emit probabilities but
unrolls every split into straight-line code, which balloons flash usage and
compile time for a 60-tree forest.

This exporter emits the forest as flat ``static const`` arrays walked by a ~20
line interpreter. It is compact, compiles in seconds, and reproduces
``predict_proba``'s class-1 column exactly, because it stores each leaf's class
distribution and averages over trees the same way scikit-learn does.

Usage
-----
    python ml_pipeline/export_model.py
    python ml_pipeline/export_model.py --axes acc_gyro      # 38-feature variant
    python ml_pipeline/export_model.py --n-estimators 30 --max-depth 8
    python ml_pipeline/export_model.py --no-write           # report only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

try:
    from . import config
    from .feature_extraction import FEATURE_NAMES, load_feature_table
    from .train_evaluate import confusion_counts, metrics_from_counts
except ImportError:  # pragma: no cover - direct script execution
    import config
    from feature_extraction import FEATURE_NAMES, load_feature_table
    from train_evaluate import confusion_counts, metrics_from_counts

HEADER_PATH = config.PROJECT_ROOT / "include" / "RandomForest.h"
FEATURES_JSON_PATH = config.PROJECT_ROOT / "include" / "features_order.json"

# Axis subsets. "all9" is the full 53-feature space the pipeline was evaluated
# on; "acc_gyro" drops every magnetometer-derived feature for 6-DoF hardware.
AXIS_SETS: dict[str, tuple[str, ...]] = {
    "all9": config.SENSOR_AXES,
    "acc_gyro": config.ACC_AXES + config.GYR_AXES,
}

LEAF_SENTINEL = 255  # feature index marking a leaf node
THRESHOLD_SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
N_SELFTEST_VECTORS = 3


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #


def select_features(axis_set: str) -> list[str]:
    """Feature names for an axis subset, preserving the pipeline's order."""
    if axis_set not in AXIS_SETS:
        raise ValueError(f"unknown axis set {axis_set!r}; choose from {list(AXIS_SETS)}")
    axes = AXIS_SETS[axis_set]

    keep: list[str] = []
    for name in FEATURE_NAMES:
        prefix = name.split("_")[0]
        if prefix in config.SENSOR_AXES:  # per-axis statistic
            if prefix in axes:
                keep.append(name)
        elif name.startswith("acc") or name.startswith("gyr"):  # magnitude / SMA
            keep.append(name)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unclassified feature name: {name}")
    return keep


# --------------------------------------------------------------------------- #
# Out-of-fold probabilities and threshold sweep
# --------------------------------------------------------------------------- #


def oof_probabilities(
    table: pd.DataFrame, feature_names: list[str], forest_params: dict
) -> np.ndarray:
    """Leave-One-Subject-Out out-of-fold P(fall) for every window.

    Thresholds must be chosen on predictions from subjects the model never saw,
    otherwise the operating point is tuned on training data.
    """
    X = table[feature_names].to_numpy(dtype=float)
    y = table["label"].to_numpy(dtype=int)
    groups = table["subject_id"].to_numpy()

    proba = np.full(len(table), np.nan, dtype=float)
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = RandomForestClassifier(
            random_state=config.RANDOM_STATE, n_jobs=-1, **forest_params
        )
        model.fit(X[train_idx], y[train_idx])
        proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    assert not np.isnan(proba).any(), "every window must receive an out-of-fold score"
    return proba


def threshold_table(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Pooled sensitivity/specificity/F1 per decision threshold."""
    rows = []
    for threshold in THRESHOLD_SWEEP:
        counts = confusion_counts(y_true, (proba >= threshold).astype(int))
        metrics = metrics_from_counts(counts)
        rows.append(
            {
                "threshold": threshold,
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "missed_falls": counts["fn"],
                "false_alarms": counts["fp"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Forest flattening
# --------------------------------------------------------------------------- #


class FlatForest:
    """Struct-of-arrays representation of a fitted RandomForestClassifier.

    Leaves are marked with ``feature == LEAF_SENTINEL`` and reuse the
    ``threshold`` slot to carry P(class = 1), which avoids a separate array.
    """

    def __init__(self, forest: RandomForestClassifier):
        self.features: list[int] = []
        self.thresholds: list[float] = []
        self.left: list[int] = []
        self.right: list[int] = []
        self.roots: list[int] = []
        self.node_counts: list[int] = []
        self.max_depth = 0

        for estimator in forest.estimators_:
            self.roots.append(len(self.features))
            self._append_tree(estimator.tree_)
            self.node_counts.append(len(self.features) - self.roots[-1])
            self.max_depth = max(self.max_depth, int(estimator.tree_.max_depth))

        if len(self.features) > 65535:
            raise ValueError(
                f"{len(self.features)} nodes exceeds the uint16 child index range; "
                "reduce --n-estimators or --max-depth"
            )

    def _append_tree(self, tree) -> None:
        base = len(self.features)
        n_nodes = tree.node_count
        # Reserve slots so children can be written as absolute global indices.
        self.features.extend([0] * n_nodes)
        self.thresholds.extend([0.0] * n_nodes)
        self.left.extend([0] * n_nodes)
        self.right.extend([0] * n_nodes)

        values = tree.value.reshape(n_nodes, -1).astype(float)
        totals = values.sum(axis=1, keepdims=True)
        # sklearn has changed whether tree_.value holds counts or fractions;
        # normalising here is correct under either convention.
        distribution = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)

        for node in range(n_nodes):
            index = base + node
            left_child = int(tree.children_left[node])
            if left_child == -1:  # leaf
                self.features[index] = LEAF_SENTINEL
                self.thresholds[index] = float(distribution[node, 1])
                self.left[index] = 0
                self.right[index] = 0
            else:
                feature_index = int(tree.feature[node])
                if feature_index >= LEAF_SENTINEL:
                    raise ValueError("feature index collides with the leaf sentinel")
                self.features[index] = feature_index
                self.thresholds[index] = float(tree.threshold[node])
                self.left[index] = base + left_child
                self.right[index] = base + int(tree.children_right[node])

    @property
    def n_nodes(self) -> int:
        return len(self.features)

    @property
    def n_trees(self) -> int:
        return len(self.roots)

    @property
    def n_leaves(self) -> int:
        return sum(1 for f in self.features if f == LEAF_SENTINEL)

    def flash_bytes(self) -> int:
        """uint8 feature + float threshold + 2 x uint16 child + root table."""
        return self.n_nodes * (1 + 4 + 2 + 2) + self.n_trees * 2

    def predict_proba(self, x: np.ndarray) -> float:
        """Reference implementation, mirrors the generated C++ exactly."""
        total = 0.0
        for root in self.roots:
            node = root
            while self.features[node] != LEAF_SENTINEL:
                node = (
                    self.left[node]
                    if x[self.features[node]] <= self.thresholds[node]
                    else self.right[node]
                )
            total += self.thresholds[node]
        return total / self.n_trees


# --------------------------------------------------------------------------- #
# C++ emission
# --------------------------------------------------------------------------- #


def _wrap(values: list[str], per_line: int, indent: str = "    ") -> str:
    lines = [
        indent + ", ".join(values[i : i + per_line]) for i in range(0, len(values), per_line)
    ]
    return ",\n".join(lines)


def _float_literal(value: float) -> str:
    """Render a C++ float literal.

    ``{:.9g}`` formats 0.0 as "0" and 1.0 as "1"; suffixing those with 'f' gives
    ``0f``, which is an integer literal with an unknown user-defined suffix and
    fails to compile. Force a decimal point when the mantissa has no '.' or 'e'.
    """
    if not np.isfinite(value):
        raise ValueError(f"cannot emit non-finite literal: {value}")
    text = f"{value:.9g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "f"


def render_header(
    flat: FlatForest,
    feature_names: list[str],
    axis_set: str,
    forest_params: dict,
    selftest: list[tuple[np.ndarray, float]],
    metadata: dict[str, object],
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    source_indices = [FEATURE_NAMES.index(name) for name in feature_names]

    feature_comment = "\n".join(
        f"//   [{i:2d}] {name}" for i, name in enumerate(feature_names)
    )
    names_array = _wrap([f'"{n}"' for n in feature_names], 4)

    selftest_blocks = []
    for index, (vector, expected) in enumerate(selftest):
        literals = _wrap([_float_literal(v) for v in vector], 6, indent="        ")
        selftest_blocks.append(
            f"    // vector {index}\n    {{\n{literals}\n    }}"
        )
    selftest_vectors = ",\n".join(selftest_blocks)
    selftest_expected = _wrap([_float_literal(p) for _, p in selftest], 4)

    metrics_comment = "\n".join(f"//   {k}: {v}" for k, v in metadata.items())

    return f"""// clang-format off
// =============================================================================
//  RandomForest.h - generated by ml_pipeline/export_model.py. DO NOT EDIT.
// =============================================================================
//  Generated : {generated}
//  Axis set  : {axis_set}
//  Features  : {len(feature_names)}
//  Trees     : {flat.n_trees} (max depth {flat.max_depth})
//  Nodes     : {flat.n_nodes} ({flat.n_leaves} leaves)
//  Flash     : ~{flat.flash_bytes() / 1024.0:.1f} KiB of const tables
//
//  Leave-One-Subject-Out performance of this configuration:
{metrics_comment}
//
//  The forest is stored as flat arrays and walked by an interpreter, so
//  RandomForest::predictProba() reproduces scikit-learn's
//  predict_proba(X)[:, 1] exactly (mean of per-leaf class distributions).
//
//  Feature vector layout expected by predict*(), see features_order.json:
{feature_comment}
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>

// Set to 0 to drop the feature-name table and the self-test vectors from flash.
#ifndef RF_INCLUDE_METADATA
#define RF_INCLUDE_METADATA 1
#endif

namespace RandomForest {{

constexpr std::size_t kFeatureCount = {len(feature_names)};
constexpr std::size_t kTreeCount = {flat.n_trees};
constexpr std::size_t kNodeCount = {flat.n_nodes};
constexpr std::uint8_t kLeafSentinel = {LEAF_SENTINEL};

// The feature extractor always produces the full 53-value pipeline vector.
// kSourceIndex maps model input i -> pipeline vector index, so a reduced model
// (e.g. the 6-DoF acc_gyro export) drops in without touching the firmware.
constexpr std::size_t kPipelineFeatureCount = {len(FEATURE_NAMES)};

static const std::uint8_t kSourceIndex[kFeatureCount] = {{
{_wrap([str(v) for v in source_indices], 20)}
}};

/// Gather the model's inputs from the full pipeline feature vector.
inline void gather(const float *pipelineFeatures, float *modelInput) {{
    for (std::size_t i = 0; i < kFeatureCount; ++i) {{
        modelInput[i] = pipelineFeatures[kSourceIndex[i]];
    }}
}}

// Split feature per node; kLeafSentinel marks a leaf.
static const std::uint8_t kFeature[kNodeCount] = {{
{_wrap([str(v) for v in flat.features], 24)}
}};

// Split threshold for internal nodes; P(class = 1) for leaves.
static const float kThreshold[kNodeCount] = {{
{_wrap([_float_literal(v) for v in flat.thresholds], 8)}
}};

static const std::uint16_t kLeft[kNodeCount] = {{
{_wrap([str(v) for v in flat.left], 20)}
}};

static const std::uint16_t kRight[kNodeCount] = {{
{_wrap([str(v) for v in flat.right], 20)}
}};

static const std::uint16_t kRoot[kTreeCount] = {{
{_wrap([str(v) for v in flat.roots], 16)}
}};

/// Walk one tree and return the leaf's P(class = 1).
inline float treeProba(std::size_t tree, const float *x) {{
    std::uint16_t node = kRoot[tree];
    while (kFeature[node] != kLeafSentinel) {{
        node = (x[kFeature[node]] <= kThreshold[node]) ? kLeft[node] : kRight[node];
    }}
    return kThreshold[node];
}}

/// Mean P(fall) over the forest - equivalent to sklearn predict_proba[:, 1].
inline float predictProba(const float *x) {{
    float total = 0.0f;
    for (std::size_t tree = 0; tree < kTreeCount; ++tree) {{
        total += treeProba(tree, x);
    }}
    return total / static_cast<float>(kTreeCount);
}}

/// Number of trees whose own leaf probability favours "fall" (majority votes).
inline std::uint16_t voteCount(const float *x) {{
    std::uint16_t votes = 0;
    for (std::size_t tree = 0; tree < kTreeCount; ++tree) {{
        if (treeProba(tree, x) >= 0.5f) {{
            ++votes;
        }}
    }}
    return votes;
}}

/// Thresholded decision. Pass the operating point chosen from the LOSO sweep.
inline bool predict(const float *x, float threshold) {{
    return predictProba(x) >= threshold;
}}

#if RF_INCLUDE_METADATA

static const char *const kFeatureNames[kFeatureCount] = {{
{names_array}
}};

constexpr std::size_t kSelfTestCount = {len(selftest)};

static const float kSelfTestInput[kSelfTestCount][kFeatureCount] = {{
{selftest_vectors}
}};

static const float kSelfTestExpected[kSelfTestCount] = {{
{selftest_expected}
}};

/// Verifies the flashed tables against probabilities computed by scikit-learn.
inline bool selfTest(float tolerance = 1e-4f) {{
    for (std::size_t i = 0; i < kSelfTestCount; ++i) {{
        const float got = predictProba(kSelfTestInput[i]);
        const float want = kSelfTestExpected[i];
        const float diff = got > want ? got - want : want - got;
        if (diff > tolerance) {{
            return false;
        }}
    }}
    return true;
}}

#endif  // RF_INCLUDE_METADATA

}}  // namespace RandomForest
// clang-format on
"""


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features-csv", type=Path, default=config.FEATURES_CSV)
    parser.add_argument("--header", type=Path, default=HEADER_PATH)
    parser.add_argument("--features-json", type=Path, default=FEATURES_JSON_PATH)
    parser.add_argument(
        "--axes",
        choices=tuple(AXIS_SETS),
        default="all9",
        help="'all9' = 53 features (9-DoF); 'acc_gyro' = 38 features (6-DoF)",
    )
    parser.add_argument("--n-estimators", type=int, default=config.RF_PARAMS["n_estimators"])
    parser.add_argument("--max-depth", type=int, default=config.RF_PARAMS["max_depth"])
    parser.add_argument(
        "--skip-loso",
        action="store_true",
        help="skip the out-of-fold threshold sweep (faster, less informative)",
    )
    parser.add_argument("--no-write", action="store_true", help="report without writing files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    table = load_feature_table(args.features_csv)
    feature_names = select_features(args.axes)
    forest_params = dict(config.RF_PARAMS)
    forest_params["n_estimators"] = args.n_estimators
    forest_params["max_depth"] = args.max_depth

    X = table[feature_names].to_numpy(dtype=float)
    y = table["label"].to_numpy(dtype=int)

    print(
        f"Axis set '{args.axes}': {len(feature_names)} features, "
        f"{len(table)} windows, {int((y == 1).sum())} positive"
    )
    print(f"Forest: {forest_params}")

    metadata: dict[str, object] = {
        "axis_set": args.axes,
        "features": len(feature_names),
        "windows": len(table),
        "subjects": int(table["subject_id"].nunique()),
    }

    sweep = None
    if not args.skip_loso:
        print("\nRunning LOSO out-of-fold probabilities for threshold selection...")
        proba = oof_probabilities(table, feature_names, forest_params)
        sweep = threshold_table(y, proba)
        print(sweep.round(4).to_string(index=False))
        default_row = sweep[np.isclose(sweep["threshold"], 0.30)].iloc[0]
        metadata.update(
            {
                "loso_threshold_0.30": (
                    f"sens={default_row['sensitivity']:.3f} "
                    f"spec={default_row['specificity']:.3f} "
                    f"f1={default_row['f1']:.3f}"
                ),
                "loso_threshold_0.50": (
                    lambda r: f"sens={r['sensitivity']:.3f} spec={r['specificity']:.3f} f1={r['f1']:.3f}"
                )(sweep[np.isclose(sweep["threshold"], 0.50)].iloc[0]),
            }
        )

    print("\nFitting the deployment forest on all subjects...")
    forest = RandomForestClassifier(
        random_state=config.RANDOM_STATE, n_jobs=-1, **forest_params
    )
    forest.fit(X, y)

    flat = FlatForest(forest)
    print(
        f"Flattened: {flat.n_trees} trees, {flat.n_nodes} nodes "
        f"({flat.n_leaves} leaves), max depth {flat.max_depth}, "
        f"~{flat.flash_bytes() / 1024.0:.1f} KiB of const tables"
    )

    # Parity against scikit-learn over the whole dataset.
    sk_proba = forest.predict_proba(X)[:, 1]
    flat_proba = np.array([flat.predict_proba(row) for row in X])
    max_diff = float(np.abs(sk_proba - flat_proba).max())
    print(f"Interpreter vs sklearn predict_proba: max |diff| = {max_diff:.3e}")
    if max_diff > 1e-6:
        raise SystemExit("flattened forest does not reproduce sklearn probabilities")

    # Self-test vectors: one confident ADL, one confident fall, one borderline.
    order = np.argsort(sk_proba)
    picks = [order[0], order[-1], order[len(order) // 2]][:N_SELFTEST_VECTORS]
    selftest = [(X[i].astype(np.float32), float(sk_proba[i])) for i in picks]
    print(f"Self-test vectors P(fall): {[round(p, 4) for _, p in selftest]}")

    if args.no_write:
        print("\n--no-write: nothing written")
        return 0

    header_text = render_header(
        flat, feature_names, args.axes, forest_params, selftest, metadata
    )
    header_path = Path(args.header)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(header_text, encoding="utf-8")
    print(f"\nWrote {header_path} ({len(header_text) / 1024.0:.0f} KiB of source)")

    features_json = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "axis_set": args.axes,
        "sample_rate_note": (
            "Training windows come from the UMAFall wrist stream at ~20 Hz; "
            "features are window statistics, so the firmware window length in "
            "seconds is what must match, not the sample count."
        ),
        "window_seconds": config.WindowConfig().window_ms / 1000.0,
        "overlap": config.WindowConfig().overlap,
        "units": {
            "accelerometer": "g (training data saturates at +/-8 g)",
            "gyroscope": "deg/s (training data saturates at +/-256 deg/s)",
            "magnetometer": (
                "uncalibrated MPU-9250 magnetometer counts scaled by the UMAFall "
                "capture; |M| median ~155 (NOT calibrated uT)"
            ),
        },
        "feature_count": len(feature_names),
        "features": [
            {"index": i, "name": name} for i, name in enumerate(feature_names)
        ],
    }
    json_path = Path(args.features_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(features_json, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")

    if sweep is not None:
        sweep_path = config.OUTPUT_DIR / "threshold_sweep.csv"
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep.to_csv(sweep_path, index=False)
        print(f"Wrote {sweep_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
