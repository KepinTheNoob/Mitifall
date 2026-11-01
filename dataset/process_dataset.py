from __future__ import annotations

import argparse
import re
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_RAW_DIR = SCRIPT_DIR / "raw"
DEFAULT_PROCESSED_DIR = SCRIPT_DIR / "processed"

METADATA_LINES = 40

WRIST_SENSOR_ID = 3

SENSOR_TYPES = {
    0: ("Ax", "Ay", "Az"),
    1: ("Gx", "Gy", "Gz"),
    2: ("Mx", "My", "Mz"),
}

OUTPUT_COLUMNS = ["TimeStamp", "Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Mx", "My", "Mz"]

RAW_COLUMNS = [
    "TimeStamp",
    "SampleNo",
    "X-Axis",
    "Y-Axis",
    "Z-Axis",
    "SensorType",
    "SensorID",
]

def read_header_line(path: Path) -> str | None:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in islice(fh, METADATA_LINES, METADATA_LINES + 1):
            return line
    return None

def header_looks_valid(header: str | None) -> bool:
    if not header:
        return False
    return "timestamp" in header.lower() and "sensor" in header.lower()

def read_raw_recording(path: Path) -> pd.DataFrame:
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

    for column in RAW_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["TimeStamp", "SensorType", "SensorID"])

def split_streams(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
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
    if other.empty:
        return base.reindex(columns=[*base.columns, *other.columns[1:]])

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
    base = streams[0].copy()
    attach = align_nearest if method == "nearest" else align_interpolate

    for sensor_type in (1, 2):
        base = attach(base, streams[sensor_type])

    return base

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
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

def classify(file_name: str) -> str | None:
    tokens = [t.lower() for t in re.split(r"[_\-.\s]+", Path(file_name).stem)]

    if "adl" in tokens:
        return "ADL"
    if "fall" in tokens:
        return "Fall"

    lowered = re.sub(r"umafall", "", Path(file_name).stem, flags=re.IGNORECASE).lower()
    if "adl" in lowered:
        return "ADL"
    if "fall" in lowered:
        return "Fall"
    return None

def process_file(path: Path, out_dir: Path, method: str) -> tuple[int, list[str]]:
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
        except Exception as exc:
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
