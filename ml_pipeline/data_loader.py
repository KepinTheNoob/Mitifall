"""Discovery and loading of the processed UMAFall recordings.

Responsibilities kept deliberately narrow: find the CSVs, work out *who*
produced each recording and *what* class it belongs to, and hand back validated
DataFrames. No feature logic lives here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:  # works both as a package and as loose scripts
    from . import config
except ImportError:  # pragma: no cover - direct script execution
    import config

# "UMAFall_Subject_01_ADL_Walking_1_2017-04-14_23-38-23.csv" -> 1
SUBJECT_RE = re.compile(r"Subject[_\-\s]*(\d+)", re.IGNORECASE)

# Movement description sits between the class token and the trial number:
# ..._ADL_Walking_1_2017-...  /  ..._Fall_backwardFall_2_2016-...
# The class token must start at a separator, otherwise the "Fall" inside the
# "UMAFall_" prefix matches and the subject number is mistaken for the trial.
ACTIVITY_RE = re.compile(
    r"(?:^|[_\-])(?:ADL|Fall)[_\-]+(?P<activity>.+?)[_\-]+(?P<trial>\d+)(?:[_\-]|$)",
    re.IGNORECASE,
)


class RecordingError(ValueError):
    """Raised when a file cannot be interpreted as a labelled recording."""


@dataclass(frozen=True)
class Recording:
    """Metadata for one experiment file (no sensor data attached)."""

    path: Path
    subject_id: int
    label: int
    class_dir: str
    activity: str
    trial: int | None

    @property
    def name(self) -> str:
        return self.path.name


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #


def parse_subject_id(file_name: str) -> int:
    """Extract the subject number, tolerating prefix variations.

    Uses ``Subject_(\\d+)`` so ``UMAFall_Subject_07_...`` and a hypothetical
    ``Subject_7_...`` both resolve to 7.
    """
    match = SUBJECT_RE.search(file_name)
    if not match:
        raise RecordingError(f"cannot parse subject id from {file_name!r}")
    return int(match.group(1))


def parse_label(path: Path) -> tuple[int, str]:
    """Resolve the binary label from the class directory, falling back to name.

    Substring matching on the full name is unsafe: every file carries the
    ``UMAFall_`` prefix, which contains "fall". The filename fallback therefore
    inspects name *tokens*.
    """
    parent = path.parent.name
    for class_dir, label in config.CLASS_DIRS.items():
        if parent.lower() == class_dir.lower():
            return label, class_dir

    tokens = [t.lower() for t in re.split(r"[_\-.\s]+", path.stem)]
    for class_dir, label in config.CLASS_DIRS.items():
        if class_dir.lower() in tokens:
            return label, class_dir

    raise RecordingError(f"cannot determine ADL/Fall label for {path.name!r}")


def parse_activity(file_name: str) -> tuple[str, int | None]:
    """Extract the movement description and trial number (best effort)."""
    match = ACTIVITY_RE.search(Path(file_name).stem)
    if not match:
        return "unknown", None
    return match.group("activity"), int(match.group("trial"))


# --------------------------------------------------------------------------- #
# Discovery / loading
# --------------------------------------------------------------------------- #


def discover_recordings(processed_dir: Path | None = None) -> list[Recording]:
    """List every labelled recording under ``dataset/processed/{ADL,Fall}/``."""
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    if not processed_dir.is_dir():
        raise FileNotFoundError(
            f"processed dataset not found: {processed_dir}. "
            "Run dataset/process_dataset.py first."
        )

    recordings: list[Recording] = []
    for class_dir in config.CLASS_DIRS:
        class_path = processed_dir / class_dir
        if not class_path.is_dir():
            continue
        for path in sorted(class_path.glob("*.csv")):
            label, resolved_dir = parse_label(path)
            activity, trial = parse_activity(path.name)
            recordings.append(
                Recording(
                    path=path,
                    subject_id=parse_subject_id(path.name),
                    label=label,
                    class_dir=resolved_dir,
                    activity=activity,
                    trial=trial,
                )
            )

    if not recordings:
        raise FileNotFoundError(f"no CSV recordings found under {processed_dir}")
    return recordings


def load_recording(recording: Recording) -> pd.DataFrame:
    """Load one recording, validated and sorted by timestamp."""
    df = pd.read_csv(recording.path)

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RecordingError(f"{recording.name}: missing columns {missing}")

    df = df[list(config.REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    if df.empty:
        raise RecordingError(f"{recording.name}: no usable numeric rows")

    return df.sort_values(config.TIME_COLUMN, kind="mergesort").reset_index(drop=True)


def summarize_recordings(recordings: list[Recording]) -> pd.DataFrame:
    """Per-subject file counts per class - useful for spotting empty LOSO folds."""
    frame = pd.DataFrame(
        [
            {"subject_id": r.subject_id, "class_dir": r.class_dir, "label": r.label}
            for r in recordings
        ]
    )
    table = (
        frame.pivot_table(
            index="subject_id", columns="class_dir", values="label", aggfunc="size"
        )
        .fillna(0)
        .astype(int)
    )
    for class_dir in config.CLASS_DIRS:
        if class_dir not in table.columns:
            table[class_dir] = 0
    return table[list(config.CLASS_DIRS)].sort_index()
