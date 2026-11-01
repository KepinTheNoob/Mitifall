from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import re
import signal
import sqlite3
import ssl
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry_db

try:
    import paho.mqtt.client as mqtt
except ImportError:
    sys.exit("paho-mqtt is not installed: pip install -r storage/requirements.txt")
if not hasattr(mqtt, "CallbackAPIVersion"):
    sys.exit("paho-mqtt 2.0 or newer is required: pip install -U 'paho-mqtt>=2.0'")

log = logging.getLogger("mitifall.logger")

AIO_HOST = "io.adafruit.com"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SECRETS = PROJECT_ROOT / "include" / "secrets.h"

FIELDS = ("accel_mag", "motion_status", "pressure", "altitude")

FEED_ROLES = {
    "accel_mag": ("--feed-accel", "accel-mag"),
    "pressure": ("--feed-pressure", "pressure"),
    "altitude": ("--feed-altitude", "altitude"),
    "fall_detected": ("--feed-fall", "fall-detected"),
    "alarm_command": ("--feed-alarm", "buzzer-command"),
    "motion_status": ("--feed-motion", "motion-status"),
}
DEFAULT_ARCHIVE_FEEDS = "fall-log,fall-tilt,device-status"

RANGES = {"accel_mag": (0.0, 64.0), "pressure": (300.0, 1100.0), "altitude": (-10000.0, 10000.0)}
TRUE_WORDS = {"1", "on", "true", "yes"}
FALSE_WORDS = {"0", "off", "false", "no"}

JSON_KEYS = {"accel_mag": "accel_mag", "accel-mag": "accel_mag", "pressure": "pressure",
             "altitude": "altitude", "motion_status": "motion_status",
             "motion-status": "motion_status", "motion": "motion_status",
             "alarm_state": "alarm_state", "alarm": "alarm_state"}

MAX_PAYLOAD_CHARS = 16384
MAX_PENDING_ROWS = 50000
RETRY_WRITE_S = 5.0
AUTH_REASON_CODES = {134, 135}

@dataclass(frozen=True)
class Message:
    topic: str
    payload: bytes
    qos: int
    retain: bool
    t_wall: float
    t_mono: float

@dataclass(frozen=True)
class Event:
    kind: str
    text: str = ""

def unwrap(text: str):
    s = text.strip()
    if s.startswith("{"):
        obj = json.loads(s)
        if not isinstance(obj, dict) or "value" not in obj:
            raise ValueError("JSON payload without a \"value\" key")
        return obj["value"]
    return s

def parse_number(value, field: str) -> float:
    v = float(str(value).strip().replace(",", "."))
    lo, hi = RANGES[field]
    if not math.isfinite(v) or not lo <= v <= hi:
        raise ValueError(f"{field} {v!r} outside {lo:g}..{hi:g}")
    return v

def parse_flag(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    word = str(value).strip().lower()
    if word in TRUE_WORDS:
        return 1
    if word in FALSE_WORDS:
        return 0
    raise ValueError(f"expected 0/1, got {str(value)[:40]!r}")

class Telemetry:

    def __init__(self, hold_s: float, alarm_s: float):
        self.hold_s, self.alarm_s = hold_s, alarm_s
        self.value = dict.fromkeys(FIELDS)
        self.at = dict.fromkeys(FIELDS)
        self.alarm_until: float | None = None

    def set(self, field: str, value, t: float) -> None:
        self.value[field], self.at[field] = value, t

    def alarm(self, on: bool, t: float, latched: bool = False) -> None:
        self.alarm_until = (math.inf if latched else t + self.alarm_s) if on else None

    def snapshot(self, t: float) -> dict:
        row = {}
        for f in FIELDS:
            at = self.at[f]
            current = at is not None and (f == "motion_status" or t - at <= self.hold_s)
            row[f] = self.value[f] if current else None
        row["alarm_state"] = int(self.alarm_until is not None and t < self.alarm_until)
        return row

def parse_json_telemetry(text: str) -> dict:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("JSON telemetry must be an object")
    out = {}
    for key, value in obj.items():
        field = JSON_KEYS.get(key)
        if field in ("motion_status", "alarm_state"):
            out[field] = parse_flag(value)
        elif field is not None:
            out[field] = parse_number(value, field)
    if not out:
        raise ValueError("JSON object has none of: " + ", ".join(sorted(set(JSON_KEYS.values()))))
    return out

def apply_message(role: str, text: str, tel: Telemetry, t: float, retained: bool) -> None:
    if role in ("accel_mag", "pressure", "altitude"):
        value = parse_number(unwrap(text), role)
        if not retained:
            tel.set(role, value, t)
    elif role == "motion_status":
        tel.set(role, parse_flag(unwrap(text)), t)
    elif role in ("fall_detected", "alarm_command"):
        on = bool(parse_flag(unwrap(text)))
        if not retained:
            tel.alarm(on, t)
    elif role == "json":
        for field, value in parse_json_telemetry(text).items():
            if field == "motion_status":
                tel.set(field, value, t)
            elif retained:
                continue
            elif field == "alarm_state":
                tel.alarm(bool(value), t, latched=True)
            else:
                tel.set(field, value, t)

def to_row(msg: Message, role: str, tel: Telemetry) -> dict | None:
    text = msg.payload.decode("utf-8", errors="backslashreplace")
    raw = {"topic": msg.topic, "payload": text[:MAX_PAYLOAD_CHARS], "qos": msg.qos,
           "retain": msg.retain, "received_at": telemetry_db.utc_timestamp(msg.t_wall).replace(" ", "T") + "Z"}
    if len(text) > MAX_PAYLOAD_CHARS:
        raw["truncated_from_chars"] = len(text)
    try:
        apply_message(role, text, tel, msg.t_mono, msg.retain)
    except (ValueError, TypeError) as exc:
        raw["parse_error"] = str(exc)[:200]
        log.warning("unparseable payload on %s: %s", msg.topic, raw["parse_error"])
    if msg.retain:
        return None
    row = tel.snapshot(msg.t_mono)
    row["timestamp"] = telemetry_db.utc_timestamp(msg.t_wall)
    row["raw_payload"] = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    return row

class Store:

    def __init__(self, path: Path):
        self.path = path
        self.conn = telemetry_db.connect(path)
        self.pending: deque[dict] = deque()
        self.written = 0
        self.retry_at = 0.0
        self.spill_path = path.with_name(f"{path.stem}.unwritten.jsonl")

    def add(self, row: dict) -> None:
        self.pending.append(row)
        if len(self.pending) > MAX_PENDING_ROWS:
            self.spill([self.pending.popleft() for _ in range(len(self.pending) // 2)])

    def flush(self, force: bool = False) -> bool:
        if not self.pending:
            return True
        if not force and time.monotonic() < self.retry_at:
            return False
        batch = list(self.pending)
        try:
            telemetry_db.insert_rows(self.conn, batch)
        except sqlite3.Error as exc:
            self.retry_at = time.monotonic() + RETRY_WRITE_S
            log.error("could not write %d row(s) to %s (%s); keeping them, retry in %.0f s",
                      len(batch), self.path, exc, RETRY_WRITE_S)
            return False
        self.pending.clear()
        self.written += len(batch)
        return True

    def spill(self, rows: list[dict]) -> None:
        with self.spill_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.error("saved %d unwritten row(s) to %s", len(rows), self.spill_path)

    def close(self) -> int:
        spilled = len(self.pending)
        if spilled:
            self.spill(list(self.pending))
            self.pending.clear()
        self.conn.close()
        return spilled

    def report_spill(self) -> None:
        try:
            with self.spill_path.open(encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
        except FileNotFoundError:
            return
        log.warning("%s holds %d row(s) an earlier run could not write to the database",
                    self.spill_path, n)

SECRET_RE = re.compile(r'^\s*#\s*define\s+(AIO_USERNAME|AIO_KEY)\s+"((?:[^"\\]|\\.)*)"', re.M)

def read_secrets(path: Path) -> dict[str, str]:
    try:
        return dict(SECRET_RE.findall(path.read_text(encoding="utf-8")))
    except OSError:
        return {}

def resolve_credentials(args) -> tuple[str, str | None, str]:
    found = read_secrets(Path(args.secrets)) if args.secrets else {}
    env = os.environ

    if args.username:
        username, user_src = args.username, "--username"
    elif env.get("AIO_USERNAME"):
        username, user_src = env["AIO_USERNAME"], "$AIO_USERNAME"
    else:
        username, user_src = found.get("AIO_USERNAME", ""), "secrets.h"

    password, key_src = None, "none (--anonymous)"
    if not args.anonymous:
        name = next((n for n in ("MQTT_PASSWORD", "AIO_KEY") if env.get(n)), None)
        if name:
            password, key_src = env[name], f"${name}"
        else:
            password, key_src = found.get("AIO_KEY") or None, "secrets.h"

    if not username and not args.prefix:
        raise SystemExit("no username: pass --username, set AIO_USERNAME or fill include/secrets.h")
    if args.host == AIO_HOST and (not username or not password):
        raise SystemExit("Adafruit IO needs AIO_USERNAME and AIO_KEY, from the environment or "
                         "include/secrets.h (cp include/secrets.example.h include/secrets.h)")
    return username, password, f"username from {user_src}, key from {key_src}"

def build_routes(args, username: str) -> dict[str, str]:
    prefix = (args.prefix or f"{username}/feeds").rstrip("/")
    routes: dict[str, str] = {}

    def add(topic: str, role: str) -> None:
        if "+" in topic or "#" in topic:
            raise SystemExit(f"wildcards are not supported: {topic}")
        if routes.get(topic, role) != role:
            raise SystemExit(f"{topic} is configured twice ({routes[topic]} and {role})")
        routes[topic] = role

    for role, (option, _default) in FEED_ROLES.items():
        feed = getattr(args, option[2:].replace("-", "_")).strip().strip("/")
        if feed:
            add(f"{prefix}/{feed}", role)
    for feed in filter(None, (f.strip().strip("/") for f in args.archive_feeds.split(","))):
        add(f"{prefix}/{feed}", "archive")
    for topic in args.json_topic or []:
        add(topic.strip(), "json")
    if args.host == AIO_HOST:
        add(f"{username}/throttle", "notice")
        add(f"{username}/errors", "notice")
    return routes

def make_client(args, username: str, password: str | None, routes: dict[str, str],
                inbox: queue.Queue) -> mqtt.Client:

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv311,
                         client_id=f"mitifall-logger-{uuid.uuid4().hex[:12]}", clean_session=True)
    if username and not args.anonymous:
        client.username_pw_set(username, password)
    if args.tls:
        client.tls_set_context(ssl.create_default_context())

    client.reconnect_delay_set(min_delay=2, max_delay=60)
    state = {"connected": False, "connected_once": False, "subscriptions": {}}

    def on_connect(c, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            if reason_code.value in AUTH_REASON_CODES and not state["connected_once"]:
                inbox.put(Event("fatal", f"broker refused the login ({reason_code}). Check the username "
                                         "and key; Adafruit IO also refuses after 20 connection attempts "
                                         "within a minute."))
            else:
                log.warning("connection refused (%s); retrying", reason_code)
            return
        state["connected"] = state["connected_once"] = True
        topics = sorted(routes)
        _, mid = c.subscribe([(t, args.qos) for t in topics])
        state["subscriptions"][mid] = topics
        inbox.put(Event("connected", f"{len(topics)} topics"))

    def on_subscribe(c, userdata, mid, reason_codes, properties):
        topics = state["subscriptions"].pop(mid, [])
        refused = [t for t, rc in zip(topics, reason_codes) if rc.is_failure]
        if refused:
            log.error("broker refused subscription to %s", ", ".join(refused))
        else:
            log.info("broker accepted all %d subscriptions", len(topics))

    def on_connect_fail(c, userdata):
        log.warning("cannot reach %s:%d; retrying with backoff", args.host, args.port)

    def on_disconnect(c, userdata, flags, reason_code, properties):
        was_connected, state["connected"] = state["connected"], False
        if was_connected and reason_code.value != 0:
            log.warning("connection lost (%s); reconnecting with backoff", reason_code)

    def on_message(c, userdata, msg):
        inbox.put(Message(msg.topic, bytes(msg.payload), msg.qos, bool(msg.retain),
                          time.time(), time.monotonic()))

    client.on_connect, client.on_subscribe = on_connect, on_subscribe
    client.on_connect_fail = on_connect_fail
    client.on_disconnect, client.on_message = on_disconnect, on_message
    return client

def drain(inbox: queue.Queue, timeout: float) -> list:
    items = []
    try:
        items.append(inbox.get(timeout=timeout) if timeout > 0 else inbox.get_nowait())
        while True:
            items.append(inbox.get_nowait())
    except queue.Empty:
        pass
    return items

def process(item, routes: dict[str, str], tel: Telemetry, store: Store, args, counters: dict):
    if isinstance(item, Event):
        if item.kind == "fatal":
            log.error(item.text)
            return 2
        log.info("connected, subscribing to %s", item.text)
        return None
    role = routes.get(item.topic)
    if role is None:
        return None
    counters["last_message"] = item.t_mono
    if role == "notice":
        log.warning("Adafruit IO %s: %s", item.topic.rsplit("/", 1)[-1],
                    item.payload.decode("utf-8", errors="replace")[:300])
        return None
    row = to_row(item, role, tel)
    if row is not None:
        store.add(row)
        if args.verbose:
            log.info("row %s  a=%s motion=%s p=%s h=%s alarm=%s  <- %s", row["timestamp"],
                     row["accel_mag"], row["motion_status"], row["pressure"], row["altitude"],
                     row["alarm_state"], item.topic)
    return None

def run(args) -> int:
    username, password, cred_source = resolve_credentials(args)
    routes = build_routes(args, username)
    store = Store(Path(args.db))
    tel = Telemetry(args.hold, args.alarm_seconds)
    inbox: queue.Queue = queue.Queue()
    stop = threading.Event()

    log.info("database %s (%d rows so far)", store.path, telemetry_db.row_count(store.conn))
    store.report_spill()
    log.info("broker %s:%d, %s, %s", args.host, args.port,
             "TLS" if args.tls else "plain MQTT (no TLS)", cred_source)
    for topic, role in sorted(routes.items(), key=lambda kv: (kv[1], kv[0])):
        log.info("  %-14s %s", role, topic)

    def request_stop(signum, _frame):
        log.info("signal %d: stopping", signum)
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    client = make_client(args, username, password, routes, inbox)
    client.connect_async(args.host, args.port, keepalive=60)
    client.loop_start()

    exit_code = 0
    counters = {"last_message": None, "written_at_status": 0}
    next_status = time.monotonic() + args.status_every
    try:
        while not stop.is_set():
            for item in drain(inbox, timeout=1.0):
                code = process(item, routes, tel, store, args, counters)
                if code is not None:
                    exit_code = code
                    stop.set()
            store.flush()
            if time.monotonic() >= next_status:
                next_status += args.status_every
                last = counters["last_message"]
                log.info("%d row(s) stored this session (+%d in the last %.0f s), %d pending, "
                         "last message %s", store.written, store.written - counters["written_at_status"],
                         args.status_every, len(store.pending),
                         "none yet" if last is None else f"{time.monotonic() - last:.0f} s ago")
                counters["written_at_status"] = store.written
    finally:
        client.disconnect()
        client.loop_stop()
        for item in drain(inbox, timeout=0):
            if isinstance(item, Message):
                process(item, routes, tel, store, args, counters)
        total = None
        if store.flush(force=True):
            try:
                total = telemetry_db.row_count(store.conn)
            except sqlite3.Error:
                pass
        if store.close() and exit_code == 0:
            exit_code = 1
        log.info("stopped: %d row(s) stored this session%s", store.written,
                 f", {total} in the database" if total is not None else "")
    return exit_code

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Log Mitifall MQTT telemetry into storage/telemetry.db.")
    p.add_argument("--host", default=AIO_HOST, help="MQTT broker (default: %(default)s)")
    p.add_argument("--port", type=int, help="default 8883 with TLS, 1883 without")
    tls = p.add_mutually_exclusive_group()
    tls.add_argument("--tls", dest="tls", action="store_true", default=None,
                     help="TLS with certificate checks (default for io.adafruit.com)")
    tls.add_argument("--no-tls", dest="tls", action="store_false", help="plain MQTT")
    p.add_argument("--username", help="MQTT username and topic owner (default: AIO_USERNAME)")
    p.add_argument("--anonymous", action="store_true", help="connect without credentials")
    p.add_argument("--secrets", default=str(DEFAULT_SECRETS),
                   help="header with AIO_USERNAME / AIO_KEY defines ('' to skip)")
    p.add_argument("--prefix", help="feed topic prefix (default: {username}/feeds)")
    for role, (option, default) in FEED_ROLES.items():
        p.add_argument(option, default=default, help=f"feed for {role} ('' disables it)")
    p.add_argument("--archive-feeds", default=DEFAULT_ARCHIVE_FEEDS,
                   help="comma-separated feeds kept in raw_payload only (default: %(default)s)")
    p.add_argument("--json-topic", action="append", metavar="TOPIC",
                   help="full topic whose payloads are JSON objects with column names as keys")
    p.add_argument("--qos", type=int, choices=(0, 1), default=0,
                   help="subscription QoS (default: %(default)s; 1 may store duplicates)")
    p.add_argument("--db", default=str(telemetry_db.DEFAULT_DB))
    p.add_argument("--hold", type=float, default=60.0,
                   help="seconds a value is carried into later rows (default: %(default)s)")
    p.add_argument("--alarm-seconds", type=float, default=0.5,
                   help="alarm length after a trigger; AlarmManager's default is 0.5 s")
    p.add_argument("--status-every", type=float, default=60.0, help="seconds between status lines")
    p.add_argument("--verbose", action="store_true", help="log every stored row")
    args = p.parse_args(argv)
    for name in ("hold", "alarm_seconds", "status_every"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.tls is None:
        args.tls = args.host == AIO_HOST
    if args.port is None:
        args.port = 8883 if args.tls else 1883
    return args

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    return run(parse_args(argv))

if __name__ == "__main__":
    sys.exit(main())
