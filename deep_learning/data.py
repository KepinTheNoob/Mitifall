from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi

from ml_pipeline import config as ml_config
from ml_pipeline import data_loader as ml_data
from ml_pipeline import feature_extraction as ml_features

from . import config

RAW_COLUMNS = config.RAW_ACC_COLUMNS + config.RAW_GYR_COLUMNS

@dataclass(frozen=True)
class InputLayout:

    use_gyro: bool = False
    use_baro: bool = False

    @property
    def raw_inputs(self) -> tuple[str, ...]:
        names = ("ax", "ay", "az")
        if self.use_gyro:
            names += ("gx", "gy", "gz")
        if self.use_baro:
            names += (config.BARO_CHANNEL,)
        return names

    @property
    def model_channels(self) -> tuple[str, ...]:
        names = config.CHANNEL_SETS["spec_gyro" if self.use_gyro else "spec"]
        if self.use_baro:
            names += (config.BARO_CHANNEL,)
        return names

@dataclass
class WindowSet:

    acc_gyr: np.ndarray
    baro: np.ndarray | None
    table: pd.DataFrame

    @property
    def labels(self) -> np.ndarray:
        return self.table["label"].to_numpy(dtype=np.int64)

    @property
    def subjects(self) -> np.ndarray:
        return self.table["subject_id"].to_numpy(dtype=np.int64)

    def raw(self, layout: InputLayout, index: np.ndarray | None = None) -> np.ndarray:
        sel = slice(None) if index is None else index
        parts = [self.acc_gyr[sel][..., :3]]
        if layout.use_gyro:
            parts.append(self.acc_gyr[sel][..., 3:6])
        if layout.use_baro:
            if self.baro is None:
                raise ValueError(
                    "barometer channel requested, but the dataset has no pressure data "
                    "(UMAFall has none). Record labelled data with [env:raw_stream], whose "
                    f"CSV carries '{config.BARO_PRESSURE_COLUMN}', first."
                )
            parts.append(self.baro[sel][..., None])
        return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)

def lowpass_coefficients() -> tuple[np.ndarray, np.ndarray]:
    return butter(
        config.LOWPASS_ORDER, config.LOWPASS_CUTOFF_HZ, btype="low", fs=config.SAMPLE_RATE_HZ
    )

def lowpass(values: np.ndarray) -> np.ndarray:
    b, a = lowpass_coefficients()
    zi = lfilter_zi(b, a)
    out = np.empty_like(values, dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j], _ = lfilter(b, a, values[:, j], zi=zi * values[0, j])
    return out

def resample_uniform(t_ms: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_ms = np.asarray(t_ms, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    uniq, inverse, counts = np.unique(t_ms, return_inverse=True, return_counts=True)
    if len(uniq) != len(t_ms):
        sums = np.zeros((len(uniq), values.shape[1]))
        np.add.at(sums, inverse, values)
        values = sums / counts[:, None]
        t_ms = uniq
    period = config.SAMPLE_PERIOD_MS
    n = int(np.floor((t_ms[-1] - t_ms[0]) / period + 1e-9)) + 1
    grid = t_ms[0] + np.arange(n) * period
    out = np.column_stack([np.interp(grid, t_ms, values[:, j]) for j in range(values.shape[1])])
    return grid, out

def pressure_to_altitude(pressure_hpa: np.ndarray, reference_hpa: float) -> np.ndarray:
    return 44330.0 * (1.0 - np.power(pressure_hpa / reference_hpa, 0.1902949))

def load_ml_feature_table() -> pd.DataFrame:
    if ml_config.FEATURES_CSV.is_file():
        table = ml_features.load_feature_table()
    else:
        table = ml_features.build_feature_table(verbose=False)
        ml_features.save_feature_table(table)
    table = table.reset_index(drop=True)

    is_fall = table["class_dir"].eq("Fall").to_numpy()
    if not np.array_equal(is_fall, table["label"].to_numpy(dtype=bool)):
        raise ValueError(
            "feature table was not built with fall_label_mode='file'; rebuild it with "
            "python ml_pipeline/feature_extraction.py"
        )
    return table

def _cache_key(table: pd.DataFrame, with_baro: bool) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(table[["source_file", "window_start_ms"]], index=False).values.tobytes())
    h.update(repr((config.SAMPLE_RATE_HZ, config.WINDOW_SAMPLES, config.LOWPASS_CUTOFF_HZ,
                   config.LOWPASS_ORDER, with_baro, RAW_COLUMNS)).encode())
    return h.hexdigest()[:16]

def _read_pressure(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    header = pd.read_csv(path, nrows=0).columns
    if config.BARO_PRESSURE_COLUMN not in header:
        return None
    frame = pd.read_csv(path, usecols=[ml_config.TIME_COLUMN, config.BARO_PRESSURE_COLUMN]).dropna()
    if frame.empty:
        return None

    series = frame.groupby(ml_config.TIME_COLUMN, sort=True)[config.BARO_PRESSURE_COLUMN].mean()
    return series.index.to_numpy(float), series.to_numpy(float)

def build_windows(table: pd.DataFrame | None = None, with_baro: bool = False,
                  use_cache: bool = True, verbose: bool = True) -> WindowSet:
    table = load_ml_feature_table() if table is None else table.reset_index(drop=True)
    key = _cache_key(table, with_baro)

    if use_cache and config.WINDOW_CACHE.is_file():
        cached = np.load(config.WINDOW_CACHE, allow_pickle=False)
        if str(cached["key"]) == key:
            baro = cached["baro"] if with_baro else None
            if verbose:
                print(f"windows: loaded {len(table)} from cache ({config.WINDOW_CACHE.name})")
            return WindowSet(cached["acc_gyr"], baro, table)

    recordings = {r.name: r for r in ml_data.discover_recordings()}
    T = config.WINDOW_SAMPLES
    acc_gyr = np.empty((len(table), T, len(RAW_COLUMNS)), dtype=np.float32)
    baro = np.empty((len(table), T), dtype=np.float32) if with_baro else None

    for name, rows in table.groupby("source_file", sort=False):
        recording = recordings[name]
        frame = ml_data.load_recording(recording)
        t = frame[ml_config.TIME_COLUMN].to_numpy(float)
        grid, values = resample_uniform(t, frame[list(RAW_COLUMNS)].to_numpy(float))
        values = lowpass(values)

        altitude = None
        if with_baro:
            pressure = _read_pressure(recording.path)
            if pressure is None:
                raise ValueError(f"{name}: no '{config.BARO_PRESSURE_COLUMN}' column; UMAFall "
                                 "has no barometer - see deep_learning/README.md")
            p_grid = np.interp(grid, *pressure)
            altitude = pressure_to_altitude(p_grid, reference_hpa=float(p_grid[0]))

        for row_index, start in zip(rows.index.to_numpy(), rows["window_start_ms"].to_numpy(float)):
            offset = (start - grid[0]) / config.SAMPLE_PERIOD_MS
            first = int(round(offset))
            if abs(offset - first) > 1e-6:
                raise ValueError(f"{name}: window start {start} ms is off the 50 Hz grid")
            segment = values[first:first + T]
            if len(segment) < T:
                segment = np.vstack([segment, np.repeat(segment[-1:], T - len(segment), axis=0)])
            acc_gyr[row_index] = segment
            if with_baro:
                seg_alt = altitude[first:first + T]
                baro[row_index] = seg_alt - seg_alt[0]

    if use_cache:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(config.WINDOW_CACHE, key=np.array(key), acc_gyr=acc_gyr,
                            baro=baro if baro is not None else np.zeros((0, T), np.float32))
    if verbose:
        print(f"windows: built {len(table)} x {T} samples @ {config.SAMPLE_RATE_HZ} Hz "
              f"from {table['source_file'].nunique()} recordings")
    return WindowSet(acc_gyr, baro, table)
