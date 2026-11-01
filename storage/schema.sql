CREATE TABLE IF NOT EXISTS telemetry_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
    accel_mag     REAL,
    motion_status INTEGER CHECK (motion_status IN (0, 1)),
    pressure      REAL,
    altitude      REAL,
    alarm_state   INTEGER CHECK (alarm_state IN (0, 1)),
    raw_payload   TEXT
);

CREATE INDEX IF NOT EXISTS idx_telemetry_logs_timestamp ON telemetry_logs (timestamp);
