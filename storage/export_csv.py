from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry_db

DEFAULT_OUT_DIR = telemetry_db.STORAGE_DIR / "exports"
RATE_WINDOW = timedelta(minutes=10)

def parse_time(text: str) -> str:
    s = text.strip().replace("T", " ")
    if s[-1:] in ("Z", "z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"not a date/time: {text!r} (use YYYY-MM-DD, optionally followed by HH:MM[:SS])") from None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

def parse_stored(ts) -> datetime | None:
    try:
        return datetime.fromisoformat(ts) if isinstance(ts, str) else None
    except ValueError:
        return None

class Stats:

    def __init__(self):
        self.rows = self.alarm_rows = self.parse_errors = 0
        self.filled = dict.fromkeys(telemetry_db.COLUMNS, 0)
        self.first = self.last = None
        self.recent: deque[datetime] = deque()

    def add(self, row: dict) -> None:
        self.rows += 1
        for column, value in row.items():
            if value is not None:
                self.filled[column] += 1
        self.alarm_rows += row["alarm_state"] == 1

        self.parse_errors += '"parse_error":' in (row["raw_payload"] or "")
        t = parse_stored(row["timestamp"])
        if t is None:
            return
        self.first = t if self.first is None else min(self.first, t)
        self.last = t if self.last is None else max(self.last, t)
        self.recent.append(t)
        while t - self.recent[0] > RATE_WINDOW:
            self.recent.popleft()

    def rate_per_min(self) -> float | None:
        if len(self.recent) < 2:
            return None
        minutes = (self.recent[-1] - self.recent[0]).total_seconds() / 60
        return (len(self.recent) - 1) / minutes if minutes > 0 else None

def export(conn: sqlite3.Connection, out: Path, columns: tuple[str, ...],
           since: str | None, until: str | None) -> Stats:
    where, params = [], []
    if since:
        where.append("timestamp >= ?")
        params.append(since)
    if until:
        where.append("timestamp < ?")
        params.append(until)
    sql = (f"SELECT {', '.join(telemetry_db.COLUMNS)} FROM {telemetry_db.TABLE}"
           + (f" WHERE {' AND '.join(where)}" if where else "") + " ORDER BY id")

    stats = Stats()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    conn.execute("BEGIN")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for values in conn.execute(sql, params):
                row = dict(zip(telemetry_db.COLUMNS, values))
                writer.writerow([row[c] for c in columns])
                stats.add(row)
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        conn.execute("COMMIT")
    return stats

def summarize(stats: Stats, out: Path, columns: tuple[str, ...], min_rows: int) -> None:
    print(f"{stats.rows} row(s) written to {out}")
    if stats.first is not None:
        span = str(stats.last - stats.first).split(".")[0]
        print(f"  time span   {stats.first:%Y-%m-%d %H:%M:%S} .. {stats.last:%Y-%m-%d %H:%M:%S} UTC ({span})")
    if stats.rows:
        print("  filled cells per column:")
        for column in (c for c in columns if c not in ("id", "timestamp")):
            n = stats.filled[column]
            print(f"    {column:<14} {n:>7}  {n / stats.rows:6.1%}")
        if stats.filled["motion_status"] == 0:
            print("    (motion_status is empty: the raw_stream firmware does not publish motion-status)")
        print(f"  rows with alarm_state = 1: {stats.alarm_rows}")
        print(f"  payloads that could not be parsed: {stats.parse_errors}")
    if min_rows <= 0:
        return
    if stats.rows >= min_rows:
        print(f"  dataset target of {min_rows} rows: met ({stats.rows} rows)")
        return
    missing, rate = min_rows - stats.rows, stats.rate_per_min()
    eta = (f"; at the recent {rate:.1f} rows/min that is about {missing / rate:.0f} more minute(s)"
           " of logging" if rate else "")
    print(f"  dataset target of {min_rows} rows: not met yet, {missing} row(s) short{eta}")

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export telemetry_logs from storage/telemetry.db to CSV.")
    p.add_argument("--db", default=str(telemetry_db.DEFAULT_DB), help="database (default: %(default)s)")
    p.add_argument("--out", type=Path, help="CSV file (default: storage/exports/telemetry_<UTC time>.csv)")
    p.add_argument("--since", type=parse_time, help="first time to include (UTC unless an offset is given)")
    p.add_argument("--until", type=parse_time, help="time to stop before (exclusive)")
    p.add_argument("--no-raw", action="store_true", help="leave out the raw_payload column")
    p.add_argument("--min-rows", type=int, default=1000,
                   help="dataset size to check (default: %(default)s; 0 skips the check)")
    args = p.parse_args(argv)
    if args.since and args.until and args.since >= args.until:
        p.error("--since must be earlier than --until")
    out = args.out or DEFAULT_OUT_DIR / f"telemetry_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
    columns = tuple(c for c in telemetry_db.COLUMNS if not (args.no_raw and c == "raw_payload"))
    try:
        conn = telemetry_db.connect(args.db, create=False)
        try:
            stats = export(conn, out, columns, args.since, args.until)
        finally:
            conn.close()
    except (OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summarize(stats, out, columns, args.min_rows)
    return 0

if __name__ == "__main__":
    sys.exit(main())
