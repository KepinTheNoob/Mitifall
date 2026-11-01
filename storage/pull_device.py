from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry_db

try:
    import serial
except ImportError:
    sys.exit("pyserial is not installed: pip install -r storage/requirements.txt")

FMT = "%Y-%m-%d %H:%M:%S.%f"


def read_dump(port: serial.Serial, timeout_s: float) -> tuple[dict, list[list[str]]]:
    port.reset_input_buffer()
    port.write(b"DBDUMP\n")
    port.flush()
    header, rows, deadline = None, [], time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line.startswith("#DBDUMP-BEGIN"):
            header = dict(kv.split("=", 1) for kv in line.split()[1:])
            rows = []
        elif line.startswith("#ROW\t") and header is not None:
            cells = line.split("\t")[1:]
            if len(cells) == 7:
                rows.append(cells)
            deadline = time.monotonic() + timeout_s
        elif line.startswith("#DBDUMP-END") and header is not None:
            return header, rows
    raise SystemExit("no complete dump from the bracelet (is it running raw_stream or dl_stream?)")


def num(cell: str):
    return float(cell) if cell not in ("", None) else None


def fill_timestamps(rows: list[dict], header: dict, pulled_at: datetime) -> int:
    anchors = {}
    for r in rows:
        if r["timestamp"] and r["boot"] not in anchors:
            anchors[r["boot"]] = (datetime.strptime(r["timestamp"], FMT), r["uptime_ms"])
    if header.get("boot") and header["boot"] not in anchors:
        anchors[header["boot"]] = (pulled_at, int(header.get("uptime_ms", 0)))
    filled = 0
    for r in rows:
        if r["timestamp"] or r["boot"] not in anchors:
            continue
        ts, up = anchors[r["boot"]]
        r["timestamp"] = (ts - timedelta(milliseconds=up - r["uptime_ms"])).strftime(FMT)[:-3]
        r["payload"]["time_estimated"] = True
        filled += 1
    return filled


def to_rows(cells_list: list[list[str]]) -> list[dict]:
    out = []
    for dev_id, ts, accel, pressure, altitude, alarm, payload in cells_list:
        try:
            meta = json.loads(payload)
        except ValueError:
            meta = {"raw": payload}
        meta["device_row_id"] = int(dev_id)
        out.append({
            "timestamp": ts or None, "accel_mag": num(accel), "motion_status": None,
            "pressure": num(pressure), "altitude": num(altitude),
            "alarm_state": int(alarm) if alarm in ("0", "1") else None,
            "payload": meta, "boot": meta.get("boot", ""),
            "uptime_ms": int(meta.get("uptime_ms", 0)),
        })
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Copy the bracelet's SQLite log into storage/telemetry.db.")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--db", default=str(telemetry_db.DEFAULT_DB))
    p.add_argument("--clear", action="store_true", help="erase the bracelet's log after a successful copy")
    p.add_argument("--timeout", type=float, default=10.0)
    args = p.parse_args(argv)

    try:
        port = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as exc:
        sys.exit(f"cannot open {args.port}: {exc} (close other serial monitors)")
    with port:
        pulled_at = datetime.now(timezone.utc).replace(tzinfo=None)
        header, cells = read_dump(port, args.timeout)
        rows = to_rows(cells)
        estimated = fill_timestamps(rows, header, pulled_at)

        conn = telemetry_db.connect(args.db)
        seen = {json.loads(r[0]).get("boot", "") + ":" + str(json.loads(r[0]).get("device_row_id"))
                for r in conn.execute(f"SELECT raw_payload FROM {telemetry_db.TABLE} "
                                      "WHERE raw_payload LIKE '%\"device_row_id\"%'")}
        new = []
        for r in rows:
            key = f"{r['boot']}:{r['payload']['device_row_id']}"
            if key in seen:
                continue
            r["raw_payload"] = json.dumps(r["payload"], separators=(",", ":"))
            new.append(r)
        if new:
            telemetry_db.insert_rows(conn, new)
        total = telemetry_db.row_count(conn)
        conn.close()

        print(f"bracelet: {len(rows)} rows, {len(new)} new, {len(rows) - len(new)} already copied")
        print(f"timestamps estimated from uptime: {estimated}")
        print(f"{args.db}: {total} rows")
        if args.clear:
            port.write(b"DBCLEAR\n")
            port.flush()
            print("bracelet log cleared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
