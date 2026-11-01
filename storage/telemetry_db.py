from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

STORAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = STORAGE_DIR / "telemetry.db"
SCHEMA_FILE = STORAGE_DIR / "schema.sql"

TABLE = "telemetry_logs"
COLUMNS = ("id", "timestamp", "accel_mag", "motion_status", "pressure", "altitude",
           "alarm_state", "raw_payload")
DATA_COLUMNS = COLUMNS[1:]

INSERT_SQL = (f"INSERT INTO {TABLE} ({', '.join(DATA_COLUMNS)}) "
              f"VALUES ({', '.join('?' * len(DATA_COLUMNS))})")

def utc_timestamp(t: float | None = None) -> str:
    dt = datetime.now(timezone.utc) if t is None else datetime.fromtimestamp(t, timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

def connect(path: Path | str = DEFAULT_DB, create: bool = True) -> sqlite3.Connection:
    path = Path(path)
    if not create and not path.is_file():
        raise FileNotFoundError(f"{path} does not exist - start storage/logger_service.py first")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA busy_timeout=30000")
    if create:
        conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    return conn

def insert_rows(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.executemany(INSERT_SQL, [tuple(row.get(c) for c in DATA_COLUMNS) for row in rows])
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise

def row_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0])
