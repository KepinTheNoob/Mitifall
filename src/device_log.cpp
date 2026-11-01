#include "device_log.h"

#include <Arduino.h>
#include <LittleFS.h>
#include <esp_random.h>
#include <sqlite3.h>
#include <sys/time.h>

#include <cstdio>
#include <cstring>
#include <ctime>

namespace DeviceLog {
namespace {

constexpr const char *kDbPath = "/littlefs/telemetry.db";
constexpr std::size_t kBatchRows = 12;
constexpr std::uint32_t kMaxRows = 20000;
constexpr std::uint32_t kTrimRows = 2000;
constexpr float kMaxFill = 0.85f;

constexpr const char *kSchema =
    "CREATE TABLE IF NOT EXISTS telemetry_logs ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,"
    "accel_mag REAL,"
    "motion_status INTEGER CHECK (motion_status IN (0, 1)),"
    "pressure REAL,"
    "altitude REAL,"
    "alarm_state INTEGER CHECK (alarm_state IN (0, 1)),"
    "raw_payload TEXT);"
    "CREATE INDEX IF NOT EXISTS idx_telemetry_logs_timestamp ON telemetry_logs (timestamp);";

constexpr const char *kInsert =
    "INSERT INTO telemetry_logs (timestamp, accel_mag, motion_status, pressure, altitude, "
    "alarm_state, raw_payload) VALUES (?, ?, NULL, ?, ?, ?, ?)";

struct Pending {
    Row row;
    char event[12];
    std::uint32_t uptimeMs;
    bool synced;
    char timestamp[24];
};

sqlite3 *gDb = nullptr;
sqlite3_stmt *gInsert = nullptr;
Print *gLog = nullptr;
Pending gPending[kBatchRows];
std::size_t gPendingCount = 0;
std::uint32_t gRows = 0;
std::uint32_t gBootId = 0;
std::uint32_t gWriteErrors = 0;

bool exec(const char *sql) {
    char *err = nullptr;
    const int rc = sqlite3_exec(gDb, sql, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        if (gLog != nullptr) {
            gLog->printf("# sqlite: %s (%s)\n", err != nullptr ? err : "error", sql);
        }
        sqlite3_free(err);
        return false;
    }
    return true;
}

std::uint32_t queryCount() {
    sqlite3_stmt *stmt = nullptr;
    std::uint32_t n = 0;
    if (sqlite3_prepare_v2(gDb, "SELECT COUNT(*) FROM telemetry_logs", -1, &stmt, nullptr) == SQLITE_OK &&
        sqlite3_step(stmt) == SQLITE_ROW) {
        n = static_cast<std::uint32_t>(sqlite3_column_int64(stmt, 0));
    }
    sqlite3_finalize(stmt);
    return n;
}

bool clockSynced(timeval &tv) {
    gettimeofday(&tv, nullptr);
    return tv.tv_sec > 1700000000;
}

void utcStamp(const timeval &tv, char *out, std::size_t size) {
    std::tm utc{};
    const std::time_t sec = tv.tv_sec;
    gmtime_r(&sec, &utc);
    std::snprintf(out, size, "%04d-%02d-%02d %02d:%02d:%02d.%03ld", utc.tm_year + 1900,
                  utc.tm_mon + 1, utc.tm_mday, utc.tm_hour, utc.tm_min, utc.tm_sec,
                  static_cast<long>(tv.tv_usec / 1000));
}

void trimIfNeeded() {
    const std::size_t total = LittleFS.totalBytes();
    const bool full = total > 0 && static_cast<float>(LittleFS.usedBytes()) / total > kMaxFill;
    if (gRows <= kMaxRows && !full) {
        return;
    }
    char sql[96];
    std::snprintf(sql, sizeof(sql),
                  "DELETE FROM telemetry_logs WHERE id IN "
                  "(SELECT id FROM telemetry_logs ORDER BY id LIMIT %lu)",
                  static_cast<unsigned long>(kTrimRows));
    if (exec(sql)) {
        gRows = queryCount();
        if (gLog != nullptr) {
            gLog->printf("# device log: storage nearly full, removed the oldest %lu rows\n",
                         static_cast<unsigned long>(kTrimRows));
        }
    }
}

}

bool begin(Print *log) {
    gLog = log;
    gBootId = esp_random();
    if (!LittleFS.begin(true, "/littlefs", 10, "spiffs")) {
        if (gLog != nullptr) {
            gLog->println("# device log: LittleFS mount failed, logging off");
        }
        return false;
    }
    sqlite3_initialize();
    if (sqlite3_open(kDbPath, &gDb) != SQLITE_OK) {
        if (gLog != nullptr) {
            gLog->printf("# device log: cannot open %s: %s\n", kDbPath, sqlite3_errmsg(gDb));
        }
        sqlite3_close(gDb);
        gDb = nullptr;
        return false;
    }
    exec("PRAGMA journal_mode=TRUNCATE");
    exec("PRAGMA synchronous=NORMAL");
    if (!exec(kSchema) ||
        sqlite3_prepare_v2(gDb, kInsert, -1, &gInsert, nullptr) != SQLITE_OK) {
        sqlite3_close(gDb);
        gDb = nullptr;
        return false;
    }
    gRows = queryCount();
    if (gLog != nullptr) {
        gLog->printf("# device log: %s, %lu rows, %u of %u KiB used\n", kDbPath,
                     static_cast<unsigned long>(gRows),
                     static_cast<unsigned>(LittleFS.usedBytes() / 1024),
                     static_cast<unsigned>(LittleFS.totalBytes() / 1024));
    }
    return true;
}

bool ready() { return gDb != nullptr; }

std::uint32_t rowCount() { return gRows + gPendingCount; }

void add(const Row &row) {
    if (gDb == nullptr) {
        return;
    }
    Pending &p = gPending[gPendingCount++];
    p.row = row;
    std::snprintf(p.event, sizeof(p.event), "%s", row.event != nullptr ? row.event : "window");
    p.row.event = p.event;
    p.uptimeMs = millis();
    timeval tv{};
    p.synced = clockSynced(tv);
    if (p.synced) {
        utcStamp(tv, p.timestamp, sizeof(p.timestamp));
    }
    if (gPendingCount >= kBatchRows || std::strcmp(p.event, "window") != 0) {
        flush();
    }
}

void flush() {
    if (gDb == nullptr || gPendingCount == 0) {
        return;
    }
    const std::uint32_t started = millis();
    bool ok = exec("BEGIN");
    for (std::size_t i = 0; ok && i < gPendingCount; ++i) {
        const Pending &p = gPending[i];
        char payload[128];
        std::snprintf(payload, sizeof(payload),
                      "{\"source\":\"device\",\"boot\":\"%08lx\",\"uptime_ms\":%lu,"
                      "\"event\":\"%s\",\"time_synced\":%s}",
                      static_cast<unsigned long>(gBootId), static_cast<unsigned long>(p.uptimeMs),
                      p.event, p.synced ? "true" : "false");
        sqlite3_reset(gInsert);
        sqlite3_clear_bindings(gInsert);
        if (p.synced) {
            sqlite3_bind_text(gInsert, 1, p.timestamp, -1, SQLITE_TRANSIENT);
        } else {
            sqlite3_bind_null(gInsert, 1);
        }
        if (p.row.hasAccel) sqlite3_bind_double(gInsert, 2, p.row.accelMag);
        if (p.row.hasPressure) sqlite3_bind_double(gInsert, 3, p.row.pressureHpa);
        if (p.row.hasAltitude) sqlite3_bind_double(gInsert, 4, p.row.altitudeM);
        sqlite3_bind_int(gInsert, 5, p.row.alarm ? 1 : 0);
        sqlite3_bind_text(gInsert, 6, payload, -1, SQLITE_TRANSIENT);
        ok = sqlite3_step(gInsert) == SQLITE_DONE;
    }
    if (ok && exec("COMMIT")) {
        gRows += gPendingCount;
        gPendingCount = 0;
    } else {
        exec("ROLLBACK");
        ++gWriteErrors;
        if (gPendingCount >= kBatchRows) {
            gPendingCount = 0;
        }
        if (gLog != nullptr) {
            gLog->printf("# device log: write failed (%s)\n", sqlite3_errmsg(gDb));
        }
        return;
    }
    trimIfNeeded();
    const std::uint32_t took = millis() - started;
    if (took > 150 && gLog != nullptr) {
        gLog->printf("# device log: slow write, %lu ms\n", static_cast<unsigned long>(took));
    }
}

void info(Print &out) {
    out.printf("#DBINFO boot=%08lx uptime_ms=%lu rows=%lu used_kib=%u total_kib=%u errors=%lu\n",
               static_cast<unsigned long>(gBootId), static_cast<unsigned long>(millis()),
               static_cast<unsigned long>(rowCount()),
               static_cast<unsigned>(LittleFS.usedBytes() / 1024),
               static_cast<unsigned>(LittleFS.totalBytes() / 1024),
               static_cast<unsigned long>(gWriteErrors));
}

void dump(Print &out) {
    flush();
    out.printf("#DBDUMP-BEGIN boot=%08lx uptime_ms=%lu rows=%lu\n",
               static_cast<unsigned long>(gBootId), static_cast<unsigned long>(millis()),
               static_cast<unsigned long>(gRows));
    if (gDb != nullptr) {
        sqlite3_stmt *stmt = nullptr;
        sqlite3_prepare_v2(gDb,
                           "SELECT id, timestamp, accel_mag, pressure, altitude, alarm_state, "
                           "raw_payload FROM telemetry_logs ORDER BY id",
                           -1, &stmt, nullptr);
        while (stmt != nullptr && sqlite3_step(stmt) == SQLITE_ROW) {
            out.print("#ROW\t");
            for (int c = 0; c < 7; ++c) {
                if (c > 0) out.print('\t');
                const unsigned char *v = sqlite3_column_text(stmt, c);
                if (v != nullptr) out.print(reinterpret_cast<const char *>(v));
            }
            out.print('\n');
        }
        sqlite3_finalize(stmt);
    }
    out.println("#DBDUMP-END");
}

void clear() {
    gPendingCount = 0;
    if (gDb != nullptr && exec("DELETE FROM telemetry_logs")) {
        exec("VACUUM");
        gRows = 0;
    }
}

}
