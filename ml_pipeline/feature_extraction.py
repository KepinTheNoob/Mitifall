"""Sliding-window segmentation and feature extraction.

Design constraints
------------------
The same features must be computable on an ESP32-C3 inside the sampling loop, so
everything here is a single-pass time-domain statistic over a short buffer: no
FFTs, no filtering, no cross-axis covariance.

Feature block per window (53 values):
  * 9 axes x {mean, std, min, max, var}                     -> 45
  * acceleration vector magnitude   {max, mean, std}        ->  3
  * angular velocity vector magnitude {max, mean, std}      ->  3
  * signal magnitude area (SMA) for accelerometer, gyro     ->  2

``std`` uses the population definition (ddof = 0) to match a naive embedded
implementation, and ``var`` is kept alongside it because the task asks for both
(they are redundant by construction; tree models are indifferent, and the scaled
linear models simply see a monotone transform).

Run directly to (re)build ``ml_pipeline/outputs/features.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config
    from .data_loader import (
        Recording,
        RecordingError,
        discover_recordings,
        load_recording,
        summarize_recordings,
    )
except ImportError:  # pragma: no cover - direct script execution
    import config
    from data_loader import (
        Recording,
        RecordingError,
        discover_recordings,
        load_recording,
        summarize_recordings,
    )

_PER_AXIS_STATS = ("mean", "std", "min", "max", "var")
_MAGNITUDE_STATS = ("max", "mean", "std")


# --------------------------------------------------------------------------- #
# Feature naming (order is part of the contract with the firmware)
# --------------------------------------------------------------------------- #


def feature_names() -> list[str]:
    """Ordered list of model input names."""
    names = [f"{axis}_{stat}" for axis in config.SENSOR_AXES for stat in _PER_AXIS_STATS]
    names += [f"acc_mag_{stat}" for stat in _MAGNITUDE_STATS]
    names += [f"gyr_mag_{stat}" for stat in _MAGNITUDE_STATS]
    names += ["acc_sma", "gyr_sma"]
    return names


FEATURE_NAMES = feature_names()


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


def window_bounds(
    timestamps: np.ndarray, window: config.WindowConfig
) -> list[tuple[float, float, int, int]]:
    """Return ``(start_ms, end_ms, lo_index, hi_index)`` for each window.

    Windows are half-open in time: ``[start, start + window_ms)``. Only complete
    windows are emitted; if the recording is shorter than one window, a single
    partial window is emitted instead when ``allow_partial`` is set.
    """
    if timestamps.size == 0:
        return []

    t_start = float(timestamps[0])
    t_end = float(timestamps[-1])
    span = t_end - t_start

    if span < window.window_ms:
        if not window.allow_partial:
            return []
        starts = np.array([t_start], dtype=float)
    else:
        # +1 keeps the last window whose end lands exactly on the final sample.
        starts = np.arange(t_start, t_end - window.window_ms + 1.0, window.step_ms)

    bounds: list[tuple[float, float, int, int]] = []
    for start in starts:
        end = start + window.window_ms
        lo = int(np.searchsorted(timestamps, start, side="left"))
        hi = int(np.searchsorted(timestamps, end, side="left"))
        if hi - lo >= window.min_samples:
            bounds.append((float(start), float(end), lo, hi))
    return bounds


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #


def _axis_stats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    mean = float(values.mean())
    var = float(values.var())  # ddof = 0
    return mean, float(np.sqrt(var)), float(values.min()), float(values.max()), var


def compute_window_features(block: np.ndarray) -> dict[str, float]:
    """Features for one window.

    ``block`` is ``(n_samples, 9)`` ordered as :data:`config.SENSOR_AXES`.
    """
    features: dict[str, float] = {}
    for index, axis in enumerate(config.SENSOR_AXES):
        mean, std, minimum, maximum, var = _axis_stats(block[:, index])
        features[f"{axis}_mean"] = mean
        features[f"{axis}_std"] = std
        features[f"{axis}_min"] = minimum
        features[f"{axis}_max"] = maximum
        features[f"{axis}_var"] = var

    acc = block[:, 0:3]
    gyr = block[:, 3:6]

    for prefix, triaxial in (("acc", acc), ("gyr", gyr)):
        magnitude = np.sqrt((triaxial**2).sum(axis=1))
        features[f"{prefix}_mag_max"] = float(magnitude.max())
        features[f"{prefix}_mag_mean"] = float(magnitude.mean())
        features[f"{prefix}_mag_std"] = float(magnitude.std())
        # Signal magnitude area: mean of the summed absolute axes over the window.
        features[f"{prefix}_sma"] = float(np.abs(triaxial).sum(axis=1).mean())

    return features


def _impact_time_ms(timestamps: np.ndarray, acc: np.ndarray) -> float:
    """Timestamp of peak acceleration magnitude (the fall impact proxy)."""
    magnitude = np.sqrt((acc**2).sum(axis=1))
    return float(timestamps[int(np.argmax(magnitude))])


def extract_recording_features(
    recording: Recording,
    df: pd.DataFrame,
    window: config.WindowConfig,
) -> list[dict[str, object]]:
    """Window one recording and compute the feature block for each window."""
    timestamps = df[config.TIME_COLUMN].to_numpy(dtype=float)
    block = df[list(config.SENSOR_AXES)].to_numpy(dtype=float)

    impact_ms: float | None = None
    if recording.label == 1 and window.fall_label_mode == "impact":
        impact_ms = _impact_time_ms(timestamps, block[:, 0:3])

    rows: list[dict[str, object]] = []
    for window_index, (start, end, lo, hi) in enumerate(window_bounds(timestamps, window)):
        label = recording.label
        if impact_ms is not None:
            overlaps_impact = (
                start <= impact_ms + window.impact_halfwidth_ms
                and end >= impact_ms - window.impact_halfwidth_ms
            )
            label = 1 if overlaps_impact else 0

        row: dict[str, object] = {
            "subject_id": recording.subject_id,
            "label": label,
            "class_dir": recording.class_dir,
            "activity": recording.activity,
            "trial": recording.trial,
            "source_file": recording.name,
            "window_index": window_index,
            "window_start_ms": start,
            "window_end_ms": end,
            "n_samples": hi - lo,
        }
        row.update(compute_window_features(block[lo:hi]))
        rows.append(row)

    return rows


# --------------------------------------------------------------------------- #
# Table builder
# --------------------------------------------------------------------------- #


def build_feature_table(
    processed_dir: Path | None = None,
    window: config.WindowConfig | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract features for the whole dataset into one tidy DataFrame."""
    window = window or config.WindowConfig()
    recordings = discover_recordings(processed_dir)

    if verbose:
        print(
            f"Found {len(recordings)} recordings "
            f"({sum(r.label == 0 for r in recordings)} ADL / "
            f"{sum(r.label == 1 for r in recordings)} Fall) "
            f"across {len({r.subject_id for r in recordings})} subjects"
        )
        print(
            f"Window: {window.window_ms} ms, step {window.step_ms} ms "
            f"({window.overlap:.0%} overlap), min {window.min_samples} samples, "
            f"fall labelling: {window.fall_label_mode}"
        )

    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for recording in recordings:
        try:
            df = load_recording(recording)
            rows.extend(extract_recording_features(recording, df, window))
        except (RecordingError, ValueError) as exc:
            failures.append(f"{recording.name}: {exc}")

    if not rows:
        raise RuntimeError("feature extraction produced no windows")

    table = pd.DataFrame(rows)
    table = table[[*config.METADATA_COLUMNS, *FEATURE_NAMES]]

    # Guard against NaN/inf leaking into the model matrix.
    finite = np.isfinite(table[FEATURE_NAMES].to_numpy(dtype=float)).all(axis=1)
    dropped = int((~finite).sum())
    table = table.loc[finite].reset_index(drop=True)

    if verbose:
        print(
            f"Extracted {len(table)} windows x {len(FEATURE_NAMES)} features "
            f"({int((table.label == 1).sum())} Fall / {int((table.label == 0).sum())} ADL)"
        )
        if dropped:
            print(f"Dropped {dropped} windows containing non-finite features")
        if failures:
            print(f"Failed to process {len(failures)} recordings:")
            for note in failures[:10]:
                print(f"  - {note}")
        print("\nPer-subject recording counts:")
        print(summarize_recordings(recordings).to_string())

    return table


def save_feature_table(
    table: pd.DataFrame,
    features_csv: Path | None = None,
    names_txt: Path | None = None,
) -> Path:
    """Persist the feature table plus the ordered feature-name contract."""
    features_csv = Path(features_csv or config.FEATURES_CSV)
    names_txt = Path(names_txt or config.FEATURE_NAMES_TXT)

    features_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(features_csv, index=False)
    names_txt.write_text("\n".join(FEATURE_NAMES) + "\n", encoding="utf-8")
    return features_csv


def load_feature_table(features_csv: Path | None = None) -> pd.DataFrame:
    """Read a previously extracted feature table."""
    features_csv = Path(features_csv or config.FEATURES_CSV)
    if not features_csv.is_file():
        raise FileNotFoundError(
            f"feature table not found: {features_csv}. Run feature_extraction.py first."
        )
    return pd.read_csv(features_csv)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def add_window_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared CLI flags so run_pipeline.py exposes the same knobs."""
    defaults = config.WindowConfig()
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=defaults.window_ms / 1000.0,
        help="sliding window duration in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=defaults.overlap,
        help="window overlap as a fraction, 0.5 = 50%% (default: %(default)s)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=defaults.min_samples,
        help="discard windows with fewer samples (default: %(default)s)",
    )
    parser.add_argument(
        "--fall-label-mode",
        choices=("file", "impact"),
        default=defaults.fall_label_mode,
        help=(
            "'file': every window of a Fall recording is positive (dataset "
            "ground truth); 'impact': only windows around the acceleration peak "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--impact-halfwidth-seconds",
        type=float,
        default=defaults.impact_halfwidth_ms / 1000.0,
        help="half-width of the positive region in 'impact' mode (default: %(default)s)",
    )


def window_config_from_args(args: argparse.Namespace) -> config.WindowConfig:
    if not 0.0 <= args.overlap < 1.0:
        raise SystemExit("--overlap must be in [0.0, 1.0)")
    if args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be positive")
    return config.WindowConfig(
        window_ms=int(round(args.window_seconds * 1000)),
        overlap=args.overlap,
        min_samples=args.min_samples,
        fall_label_mode=args.fall_label_mode,
        impact_halfwidth_ms=int(round(args.impact_halfwidth_seconds * 1000)),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract windowed features from the processed dataset.")
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--features-csv", type=Path, default=config.FEATURES_CSV)
    add_window_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    window = window_config_from_args(args)

    table = build_feature_table(args.processed_dir, window)
    path = save_feature_table(table, args.features_csv)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
