"""Process the raw UMAFall CSV recordings into aligned, per-experiment CSV files.

Pipeline
--------
1. Read every raw CSV in ``dataset/raw/`` (already unzipped).
2. Skip the 40 metadata lines so parsing starts at the header line (line 41).
3. Keep only the wrist sensor (``Sensor ID == 3``).
4. Split the interleaved sensor streams by ``Sensor Type``
   (0 = accelerometer, 1 = gyroscope, 2 = magnetometer) and align them on
   ``TimeStamp`` (nearest-timestamp match or linear interpolation).
5. Write ``[TimeStamp, Ax, Ay, Az, Gx, Gy, Gz, Mx, My, Mz]`` to
   ``dataset/processed/ADL/`` or ``dataset/processed/Fall/`` depending on the
   input file name.

Usage
-----
    python process_dataset.py                    # nearest-timestamp alignment
    python process_dataset.py --method interpolate
    python process_dataset.py --raw-dir ... --out-dir ...
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_RAW_DIR = SCRIPT_DIR / "raw"
DEFAULT_PROCESSED_DIR = SCRIPT_DIR / "processed"

# Number of metadata lines before the header line (header is line 41, 1-based).
METADATA_LINES = 40

WRIST_SENSOR_ID = 3

SENSOR_TYPES = {
    0: ("Ax", "Ay", "Az"),  # accelerometer
    1: ("Gx", "Gy", "Gz"),  # gyroscope
    2: ("Mx", "My", "Mz"),  # magnetometer
}

OUTPUT_COLUMNS = ["TimeStamp", "Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Mx", "My", "Mz"]

# Canonical names for the seven raw columns, in file order.
RAW_COLUMNS = [
    "TimeStamp",
    "SampleNo",
    "X-Axis",
    "Y-Axis",
    "Z-Axis",
    "SensorType",
    "SensorID",
]


# --------------------------------------------------------------------------- #
# Raw file parsing
# --------------------------------------------------------------------------- #


def read_header_line(path: Path) -> str | None:
    """Return the header line (line 41) of a raw recording, or ``None``."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in islice(fh, METADATA_LINES, METADATA_LINES + 1):
            return line
    return None


def header_looks_valid(header: str | None) -> bool:
    """Cheap sanity check that line 41 really is the column header."""
    if not header:
        return False
    return "timestamp" in header.lower() and "sensor" in header.lower()


def read_raw_recording(path: Path) -> pd.DataFrame:
    """Read a raw recording into a DataFrame with :data:`RAW_COLUMNS`.

    The first 40 lines are metadata; line 41 is the (commented-out) header, so
    the numeric data starts on line 42. The header is validated rather than
    parsed, because it carries a trailing ``;`` that the data rows do not have.
    """
    if not header_looks_valid(read_header_line(path)):
        raise ValueError(f"unexpected header on line {METADATA_LINES + 1}")

    df = pd.read_csv(
        path,
        sep=";",
        comment="%",
        header=None,
        names=RAW_COLUMNS,
        usecols=range(len(RAW_COLUMNS)),
        skiprows=METADATA_LINES + 1,
        skip_blank_lines=True,
        engine="c",
    )

    # Coerce everything to numbers; malformed rows become NaN and are dropped.
    for column in RAW_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["TimeStamp", "SensorType", "SensorID"])


# --------------------------------------------------------------------------- #
# Sensor alignment
# --------------------------------------------------------------------------- #


def split_streams(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Split wrist rows into one tidy frame per sensor type."""
    streams: dict[int, pd.DataFrame] = {}
    for sensor_type, axis_names in SENSOR_TYPES.items():
        stream = df.loc[df["SensorType"] == sensor_type, ["TimeStamp", "X-Axis", "Y-Axis", "Z-Axis"]]
        if stream.empty:
            streams[sensor_type] = pd.DataFrame(columns=["TimeStamp", *axis_names])
            continue

        stream = stream.rename(
            columns=dict(zip(["X-Axis", "Y-Axis", "Z-Axis"], axis_names))
        )
        stream = stream.sort_values("TimeStamp", kind="mergesort").reset_index(drop=True)
        streams[sensor_type] = stream
    return streams


def align_nearest(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Attach ``other`` to ``base`` using nearest-timestamp matching."""
    if other.empty:
        return base.reindex(columns=[*base.columns, *other.columns[1:]])

    # merge_asof needs a unique key on the right-hand side; duplicated
    # timestamps within one stream are averaged so the match is deterministic.
    right = other
    if right["TimeStamp"].duplicated().any():
        right = right.groupby("TimeStamp", as_index=False).mean()

    return pd.merge_asof(
        base,
        right,
        on="TimeStamp",
        direction="nearest",
    )


def align_interpolate(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Attach ``other`` to ``base`` by linear interpolation over time."""
    axis_names = list(other.columns[1:])
    if other.empty:
        return base.reindex(columns=[*base.columns, *axis_names])

    source = other
    if source["TimeStamp"].duplicated().any():
        source = source.groupby("TimeStamp", as_index=False).mean()

    target_times = base["TimeStamp"].to_numpy(dtype=float)
    source_times = source["TimeStamp"].to_numpy(dtype=float)

    merged = base.copy()
    for axis in axis_names:
        merged[axis] = np.interp(
            target_times,
            source_times,
            source[axis].to_numpy(dtype=float),
        )
    return merged


def align_streams(streams: dict[int, pd.DataFrame], method: str) -> pd.DataFrame:
    """Merge accelerometer/gyroscope/magnetometer onto the accelerometer clock."""
    base = streams[0].copy()  # accelerometer defines the output timeline
    attach = align_nearest if method == "nearest" else align_interpolate

    for sensor_type in (1, 2):
        base = attach(base, streams[sensor_type])

    return base


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate interior gaps, then pad the edges."""
    value_columns = [c for c in OUTPUT_COLUMNS if c != "TimeStamp"]
    present = [c for c in value_columns if df[c].notna().any()]
    if present:
        df[present] = (
            df[present]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
    return df


# --------------------------------------------------------------------------- #
# Per-file processing
# --------------------------------------------------------------------------- #


def classify(file_name: str) -> str | None:
    """Return ``"Fall"``, ``"ADL"`` or ``None`` based on the file name.

    Matching is done on name *tokens* rather than plain substrings, because the
    dataset prefix ``UMAFall_`` also contains the word "fall":
    ``UMAFall_Subject_01_ADL_Aplausing_1_...csv`` is an ADL recording.
    """
    tokens = [t.lower() for t in re.split(r"[_\-.\s]+", Path(file_name).stem)]

    if "adl" in tokens:
        return "ADL"
    if "fall" in tokens:
        return "Fall"

    # Fallback: substring search, once the dataset prefix is removed.
    lowered = re.sub(r"umafall", "", Path(file_name).stem, flags=re.IGNORECASE).lower()
    if "adl" in lowered:
        return "ADL"
    if "fall" in lowered:
        return "Fall"
    return None


def process_file(path: Path, out_dir: Path, method: str) -> tuple[int, list[str]]:
    """Process one raw recording. Returns (rows written, warnings)."""
    warnings: list[str] = []

    raw = read_raw_recording(path)
    wrist = raw[raw["SensorID"] == WRIST_SENSOR_ID]
    if wrist.empty:
        raise ValueError(f"no rows for wrist sensor (Sensor ID == {WRIST_SENSOR_ID})")

    streams = split_streams(wrist)
    if streams[0].empty:
        raise ValueError("no wrist accelerometer samples to build the timeline on")

    for sensor_type, axis_names in SENSOR_TYPES.items():
        if streams[sensor_type].empty:
            warnings.append(f"missing sensor type {sensor_type} ({'/'.join(axis_names)})")

    merged = align_streams(streams, method)
    merged = merged.reindex(columns=OUTPUT_COLUMNS)
    merged = fill_missing(merged)
    merged = merged.sort_values("TimeStamp", kind="mergesort").reset_index(drop=True)
    merged["TimeStamp"] = merged["TimeStamp"].astype("int64")

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / path.name, index=False)

    return len(merged), warnings


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"directory with the unzipped raw CSVs (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"output root; ADL/ and Fall/ are created inside (default: {DEFAULT_PROCESSED_DIR})",
    )
    parser.add_argument(
        "--method",
        choices=("nearest", "interpolate"),
        default="nearest",
        help="how to align gyroscope/magnetometer onto the accelerometer clock",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print the final summary",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    raw_dir: Path = args.raw_dir
    if not raw_dir.is_dir():
        print(f"Raw directory not found: {raw_dir}", file=sys.stderr)
        return 1

    raw_files = sorted(p for p in raw_dir.glob("*.csv") if p.is_file())
    if not raw_files:
        print(f"No CSV files found in {raw_dir}", file=sys.stderr)
        return 1

    out_dirs = {label: args.out_dir / label for label in ("ADL", "Fall")}
    for out_dir in out_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"ADL": 0, "Fall": 0}
    rows = {"ADL": 0, "Fall": 0}
    skipped: list[str] = []
    failed: list[str] = []
    warned: list[str] = []

    print(f"Processing {len(raw_files)} raw files from {raw_dir} (method: {args.method})")

    for path in raw_files:
        label = classify(path.name)
        if label is None:
            skipped.append(path.name)
            continue

        try:
            n_rows, warnings = process_file(path, out_dirs[label], args.method)
        except Exception as exc:  # noqa: BLE001 - keep going through the dataset
            failed.append(f"{path.name}: {exc}")
            continue

        counts[label] += 1
        rows[label] += n_rows

        for warning in warnings:
            warned.append(f"{path.name}: {warning}")

        if not args.quiet:
            print(f"  [{label}] {path.name} -> {n_rows} rows")

    print("\nDone.")
    print(f"  ADL  files written: {counts['ADL']:4d} ({rows['ADL']} rows) -> {out_dirs['ADL']}")
    print(f"  Fall files written: {counts['Fall']:4d} ({rows['Fall']} rows) -> {out_dirs['Fall']}")
    print(f"  Columns: {', '.join(OUTPUT_COLUMNS)}")

    if skipped:
        print(f"  Skipped (no ADL/Fall in name): {len(skipped)}")
        for name in skipped[:10]:
            print(f"    - {name}")
    if warned:
        print(f"  Warnings: {len(warned)}")
        for note in warned[:10]:
            print(f"    - {note}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for note in failed[:10]:
            print(f"    - {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
