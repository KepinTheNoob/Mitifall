from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch

from deep_learning import config
from deep_learning.data import InputLayout, build_windows
from deep_learning.metrics import threshold_free
from deep_learning.model import ProbabilityModel
from deep_learning.train import load_checkpoint

INPUT_NAME = "window"
OUTPUT_NAME = "p_fall"
CALIBRATION_WINDOWS = 512
UNITS = {"ax": "g", "ay": "g", "az": "g", "gx": "deg/s", "gy": "deg/s", "gz": "deg/s",
         config.BARO_CHANNEL: "m"}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def export_fp32(model: ProbabilityModel, n_raw: int, path: Path, opset: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)

    example = (torch.zeros(4, n_raw, config.WINDOW_SAMPLES),)
    try:
        torch.onnx.export(model, example, str(path), dynamo=True, external_data=False,
                          input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                          dynamic_shapes={"raw": {0: torch.export.Dim("batch", min=1, max=65536)}},
                          opset_version=opset)
        return "torch.export (dynamo)"
    except Exception as exc:
        print(f"dynamo exporter failed ({type(exc).__name__}: {exc}); "
              "falling back to the TorchScript exporter")
    torch.onnx.export(model, example, str(path), dynamo=False, input_names=[INPUT_NAME],
                      output_names=[OUTPUT_NAME],
                      dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
                      opset_version=opset)
    return "TorchScript"

def run_onnx(session: ort.InferenceSession, x: np.ndarray, batch: int = 512) -> np.ndarray:
    out = [session.run([OUTPUT_NAME], {INPUT_NAME: x[i:i + batch]})[0] for i in range(0, len(x), batch)]
    return np.concatenate(out).reshape(-1)

def latency_ms(session: ort.InferenceSession, x1: np.ndarray, runs: int = 300) -> float:
    for _ in range(20):
        session.run([OUTPUT_NAME], {INPUT_NAME: x1})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run([OUTPUT_NAME], {INPUT_NAME: x1})
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000.0)

def session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

def quantize_int8(fp32_path: Path, int8_path: Path, calibration: np.ndarray) -> None:
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, x: np.ndarray):
            self._items = iter(x[i:i + 1] for i in range(len(x)))

        def get_next(self):
            item = next(self._items, None)
            return None if item is None else {INPUT_NAME: item}

    pre_path = int8_path.with_suffix(".pre.onnx")
    source = fp32_path
    try:
        quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=True)
        source = pre_path
    except Exception as exc:
        print(f"quant_pre_process skipped ({type(exc).__name__}: {exc})")
    try:
        quantize_static(str(source), str(int8_path), Reader(calibration),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    finally:
        pre_path.unlink(missing_ok=True)

def benchmark_summary(meta: dict) -> dict | None:
    from deep_learning.evaluate_benchmark import run_tag

    tag = run_tag(argparse.Namespace(
        arch=meta["arch"], with_gyro=meta["use_gyro"], with_baro=meta["use_baro"],
        loss=meta["train_config"]["loss"], no_augment=not meta["augment_config"]["enabled"],
        epochs=meta["train_config"]["epochs"], seed=config.RANDOM_SEED,
        ensemble=meta.get("ensemble", 1)))
    metrics = config.OUTPUT_DIR / "benchmark" / tag / "metrics.csv"
    if not metrics.is_file():
        return None
    rows = pd.read_csv(metrics).to_dict(orient="records")
    return {"tag": tag, "report": str((metrics.parent / "report.md").relative_to(config.PROJECT_ROOT)),
            "protocol": "LOSO, 19 folds, 15 train / 3 validation subjects per fold, pooled test windows",
            "rows": rows}

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the deep-learning fall detector to ONNX.")
    p.add_argument("--checkpoint", type=Path, default=config.FINAL_CHECKPOINT)
    p.add_argument("--out", type=Path, default=config.ONNX_MODEL)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--int8", action="store_true", help="also write a static INT8 (QDQ) model")
    p.add_argument("--int8-out", type=Path, default=config.ONNX_INT8_MODEL)
    p.add_argument("--card", type=Path, default=config.MODEL_CARD)
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        raise SystemExit(f"{args.checkpoint} not found - run python deep_learning/train.py first")
    net, meta = load_checkpoint(args.checkpoint)
    prob = ProbabilityModel(net).eval()
    layout = InputLayout(use_gyro=meta["use_gyro"], use_baro=meta["use_baro"])
    n_raw = len(layout.raw_inputs)
    threshold = round(float(meta["threshold"]), 4)

    exporter = export_fp32(prob, n_raw, args.out, args.opset)
    graph = onnx.load(str(args.out))
    onnx.checker.check_model(graph)
    opset = max(o.version for o in graph.opset_import if o.domain in ("", "ai.onnx"))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KiB, opset {opset}, {exporter})")

    windows = build_windows(with_baro=layout.use_baro, verbose=False)
    x = np.ascontiguousarray(windows.raw(layout).transpose(0, 2, 1))
    y = windows.labels
    with torch.no_grad():
        p_torch = np.concatenate([prob(torch.from_numpy(x[i:i + 512])).numpy()
                                  for i in range(0, len(x), 512)])
    sess = session(args.out)
    p_onnx = run_onnx(sess, x)
    p_onnx_b1 = np.concatenate([run_onnx(sess, x[i:i + 1]) for i in range(64)])
    parity = {
        "windows": int(len(x)),
        "max_abs_diff": float(np.abs(p_torch - p_onnx).max()),
        "max_abs_diff_batch1": float(np.abs(p_torch[:64] - p_onnx_b1).max()),
        "decision_agreement": float(np.mean((p_torch >= threshold) == (p_onnx >= threshold))),
        "latency_ms_batch1_1thread": latency_ms(sess, x[:1]),
    }
    print(f"onnxruntime vs torch: max |dp| {parity['max_abs_diff']:.2e} "
          f"(batch 1: {parity['max_abs_diff_batch1']:.2e}), decisions agree "
          f"{parity['decision_agreement'] * 100:.3f} %, {parity['latency_ms_batch1_1thread']:.2f} ms/window")
    if parity["max_abs_diff"] > 1e-4 or parity["max_abs_diff_batch1"] > 1e-4:
        raise SystemExit("ONNX output does not match PyTorch - not writing a model card")

    int8 = None
    if args.int8:
        rng = np.random.default_rng(config.RANDOM_SEED)
        calib = x[rng.choice(len(x), size=min(CALIBRATION_WINDOWS, len(x)), replace=False)]
        quantize_int8(args.out, args.int8_out, calib)
        sess8 = session(args.int8_out)
        p8 = run_onnx(sess8, x)
        int8 = {
            "file": args.int8_out.name,
            "sha256": sha256(args.int8_out),
            "bytes": args.int8_out.stat().st_size,
            "quantization": "static QDQ, per-channel int8 weights, int8 activations; "
                            "Conv/MatMul/Gemm only",
            "calibration_windows": int(len(calib)),
            "max_abs_diff_vs_fp32": float(np.abs(p8 - p_onnx).max()),
            "mean_abs_diff_vs_fp32": float(np.abs(p8 - p_onnx).mean()),
            "decision_agreement_vs_fp32": float(np.mean((p8 >= threshold) == (p_onnx >= threshold))),
            "roc_auc_in_sample_fp32": threshold_free(y, p_onnx)["roc_auc"],
            "roc_auc_in_sample_int8": threshold_free(y, p8)["roc_auc"],
            "latency_ms_batch1_1thread": latency_ms(sess8, x[:1]),
        }
        print(f"wrote {args.int8_out} ({int8['bytes'] / 1024:.1f} KiB): decisions agree with fp32 "
              f"{int8['decision_agreement_vs_fp32'] * 100:.2f} %, max |dp| {int8['max_abs_diff_vs_fp32']:.3f}, "
              f"{int8['latency_ms_batch1_1thread']:.2f} ms/window")

    card = {
        "name": "mitifall_fall_dl",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "onnx": {"file": args.out.name, "sha256": sha256(args.out), "bytes": args.out.stat().st_size,
                 "opset": opset, "exporter": exporter},
        "onnx_int8": int8,
        "input": {
            "name": INPUT_NAME, "dtype": "float32", "shape": ["batch", n_raw, config.WINDOW_SAMPLES],
            "channels": list(layout.raw_inputs),
            "units": {c: UNITS[c] for c in layout.raw_inputs},
        },
        "output": {"name": OUTPUT_NAME, "dtype": "float32", "shape": ["batch"],
                   "meaning": "probability that the 2.0 s window belongs to a fall recording"},
        "preprocessing": {
            "sample_rate_hz": config.SAMPLE_RATE_HZ,
            "resampling": "linear interpolation onto a uniform 20 ms grid starting at the first sample",
            "lowpass": {"type": "butterworth", "order": config.LOWPASS_ORDER,
                        "cutoff_hz": config.LOWPASS_CUTOFF_HZ, "causal": True,
                        "initial_state": "steady state at the first sample (scipy lfilter_zi)",
                        "applies_to": [c for c in layout.raw_inputs if c != config.BARO_CHANNEL]},
            "window_samples": config.WINDOW_SAMPLES,
            "hop_samples": config.HOP_SAMPLES,
            "dh_m": "barometric altitude minus its value at the window start (not low-passed)"
                    if layout.use_baro else None,
            "derived_in_graph": ["acc_mag = |a|", "tilt_deg = acos(az / |a|)", "robust scaler"],
        },
        "decision": {"threshold": threshold,
                     "source": "Youden's J on validation subjects "
                               + ", ".join(map(str, meta["validation_subjects"]))},
        "model": {"arch": meta["arch"], "ensemble": meta.get("ensemble", 1),
                  "model_channels": meta["model_channels"], "parameters": meta["parameters"],
                  "macs_per_window": meta["macs_per_window"], "config": meta["model_config"]},
        "training": {"dataset": "UMAFall wrist (via ml_pipeline feature table, file-level labels)",
                     "validation_subjects": meta["validation_subjects"],
                     "validation_metrics": meta["validation_metrics"],
                     "best_epoch": meta["best_epoch"], "train_config": meta["train_config"],
                     "augment_config": meta["augment_config"], "seeds": meta.get("seeds")},
        "benchmark": benchmark_summary(meta),
        "parity": parity,
        "limitations": [
            "Trained on UMAFall wrist data resampled from ~20 Hz; not yet validated on Mitifall "
            "hardware recordings.",
            "Labels are file-level: every window of a fall recording counts as Fall.",
            "No barometer channel unless trained with --with-baro on recordings that have pressure.",
        ],
    }
    args.card.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.card}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
