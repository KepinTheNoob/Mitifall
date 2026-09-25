"""End-to-end runner: processed CSVs -> windowed features -> LOSO benchmark.

    python ml_pipeline/run_pipeline.py                      # extract + evaluate
    python ml_pipeline/run_pipeline.py --reuse-features      # skip extraction
    python ml_pipeline/run_pipeline.py --window-seconds 2.5 --overlap 0.5
    python ml_pipeline/run_pipeline.py --models random_forest svm_rbf
    python ml_pipeline/run_pipeline.py --fall-label-mode impact
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from . import config
    from .feature_extraction import (
        add_window_arguments,
        build_feature_table,
        load_feature_table,
        save_feature_table,
        window_config_from_args,
    )
    from .train_evaluate import evaluate
except ImportError:  # pragma: no cover - direct script execution
    import config
    from feature_extraction import (
        add_window_arguments,
        build_feature_table,
        load_feature_table,
        save_feature_table,
        window_config_from_args,
    )
    from train_evaluate import evaluate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Mitifall fall-detection training/evaluation pipeline."
    )
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=config.FEATURES_CSV,
        help="where the windowed feature table is written/read",
    )
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="reuse an existing feature table instead of re-extracting",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="only build the feature table",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(config.MODEL_NAMES),
        choices=list(config.MODEL_NAMES),
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    add_window_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    print("=" * 78)
    print("STEP 1/2  Sliding-window segmentation and feature extraction")
    print("=" * 78)

    if args.reuse_features and Path(args.features_csv).is_file():
        table = load_feature_table(args.features_csv)
        print(f"Reusing {args.features_csv} ({len(table)} windows)")
    else:
        if args.reuse_features:
            print(f"{args.features_csv} not found - extracting from scratch")
        window = window_config_from_args(args)
        table = build_feature_table(args.processed_dir, window)
        path = save_feature_table(table, args.features_csv)
        print(f"Wrote {path}")

    if args.skip_evaluation:
        print("\nSkipping evaluation (--skip-evaluation)")
        return 0

    print("\n" + "=" * 78)
    print("STEP 2/2  Leave-One-Subject-Out benchmark")
    print("=" * 78)

    evaluate(
        table,
        model_names=tuple(args.models),
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )

    print(f"\nTotal pipeline time: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
