"""SQLite persistence.

A version bundles, atomically: the per-factory corrections, the fitted
model and the per-factory LOFO errors.  The `meta.current_version`
pointer is flipped as the last statement of the same transaction, so
readers (predictions) always see one consistent committed version --
including while a recomputation is still in progress.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    id          INTEGER PRIMARY KEY,
    lambda      REAL NOT NULL,
    scores_json TEXT NOT NULL,
    model_json  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS factory_stats (
    version_id  INTEGER NOT NULL REFERENCES versions(id),
    factory_id  TEXT    NOT NULL,
    n           INTEGER NOT NULL,
    sums_json   TEXT    NOT NULL,   -- [sum speed, sum pressure, sum power]
    outer_json  TEXT    NOT NULL,   -- 3x3 outer-product sums
    correction  TEXT    NOT NULL,   -- integer zero-point correction
    PRIMARY KEY (version_id, factory_id)
);
CREATE TABLE IF NOT EXISTS factory_errors (
    version_id  INTEGER NOT NULL REFERENCES versions(id),
    factory_id  TEXT    NOT NULL,
    mse         REAL    NOT NULL,
    PRIMARY KEY (version_id, factory_id)
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Pilot batches pin the model version / correction / power offset they were
-- opened with.  These columns never change once the row exists.
CREATE TABLE IF NOT EXISTS pilot_batches (
    factory_id  TEXT    NOT NULL,
    batch_id    TEXT    NOT NULL,
    version_id  INTEGER NOT NULL REFERENCES versions(id),
    correction  TEXT    NOT NULL,   -- integer zero-point correction pin
    offset      REAL    NOT NULL,   -- power calibration offset pin
    opened_at   TEXT    NOT NULL,
    PRIMARY KEY (factory_id, batch_id)
);
-- One (replaceable) trial-run summary per batch, scored with the batch pins.
CREATE TABLE IF NOT EXISTS pilot_reports (
    factory_id   TEXT    NOT NULL,
    batch_id     TEXT    NOT NULL,
    n            INTEGER NOT NULL,
    sums_json    TEXT    NOT NULL,
    outer_json   TEXT    NOT NULL,
    mse_before   REAL    NOT NULL,
    mse_after    REAL    NOT NULL,
    reported_at  TEXT    NOT NULL,
    PRIMARY KEY (factory_id, batch_id)
);
-- Running totals of the *current* summary of every batch of a factory (raw
-- n and sums; the offset is re-derived from them under whatever model is
-- pinned).  Opening a batch reads this one row; it never scans reports.
CREATE TABLE IF NOT EXISTS report_totals (
    factory_id  TEXT PRIMARY KEY,
    n           INTEGER NOT NULL,
    sums_json   TEXT    NOT NULL
);
"""


class Conflict(Exception):
    """The version an update was based on is no longer current."""


class NoCurrentVersion(Exception):
    """No model version has been published yet."""


class UnknownFactory(Exception):
    """The factory has no correction in the version being pinned."""


class UnknownBatch(Exception):
    """No pilot batch exists for this (factory_id, batch_id)."""


def _jints(values):
    """Serialise (possibly huge) integers as strings, losslessly."""
    return json.dumps([str(v) for v in values])


def _jint_matrix(rows):
    return json.dumps([[str(v) for v in row] for row in rows])


def _parse_ints(text):
    return [int(v) for v in json.loads(text)]


def _parse_matrix(text):
    return [[int(v) for v in row] for row in json.loads(text)]


class Store:
    def __init__(self, path: str):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    def close(self):
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------- reads
    def current_version_id(self):
        with self._lock:
            return self._current_version_id(self._conn)

    @staticmethod
    def _current_version_id(conn):
        row = conn.execute(
            "SELECT value FROM meta WHERE key='current_version'"
        ).fetchone()
        return int(row["value"]) if row else None

    def get_version(self, version_id: int):
        with self._lock:
            v = self._conn.execute(
                "SELECT * FROM versions WHERE id=?", (version_id,)
            ).fetchone()
            if v is None:
                return None
            stats = self._conn.execute(
                "SELECT * FROM factory_stats WHERE version_id=? ORDER BY factory_id",
                (version_id,),
            ).fetchall()
            errors = self._conn.execute(
                "SELECT factory_id, mse FROM factory_errors WHERE version_id=?",
                (version_id,),
            ).fetchall()
        return self._assemble(v, stats, errors)

    def get_current(self):
        vid = self.current_version_id()
        return None if vid is None else self.get_version(vid)

    @staticmethod
    def _assemble(v, stats, errors):
        return {
            "version": v["id"],
            "lambda": v["lambda"],
            "lambda_scores": {float(k): val for k, val in
                              json.loads(v["scores_json"]).items()},
            "model": json.loads(v["model_json"]),
            "created_at": v["created_at"],
            "factories": [
                {
                    "factory_id": s["factory_id"],
                    "n": s["n"],
                    "sums": _parse_ints(s["sums_json"]),
                    "outer": _parse_matrix(s["outer_json"]),
                    "correction": int(s["correction"]),
                }
                for s in stats
            ],
            "factory_errors": {e["factory_id"]: e["mse"] for e in errors},
        }

    # ---------------------------------------------------------- writes
    def create_version(self, factories, result) -> int:
        """Store a freshly trained version and make it current, atomically."""
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute("BEGIN IMMEDIATE")
                new_id = self._insert_version(cur, factories, result)
                self._set_current(cur, new_id)
                self._conn.commit()
                return new_id
            except Exception:
                self._conn.rollback()
                raise

    def replace_correction(self, base_version_id: int, factories,
                           result) -> int:
        """Store a new version (one factory's correction replaced).

        `factories` is the full factory list of the base version with the
        new correction already applied.  Raises Conflict if base_version_id
        is not the current version; on any failure the current version is
        left untouched.
        """
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute("BEGIN IMMEDIATE")
                row = cur.execute(
                    "SELECT value FROM meta WHERE key='current_version'"
                ).fetchone()
                current = int(row["value"]) if row else None
                if current != base_version_id:
                    raise Conflict(
                        f"version {base_version_id} is not current "
                        f"(current: {current})"
                    )
                new_id = self._insert_version(cur, factories, result)
                self._set_current(cur, new_id)
                self._conn.commit()
                return new_id
            except Exception:
                self._conn.rollback()
                raise

    # ---------------------------------------------------------- pilot batches
    def get_batch(self, factory_id: str, batch_id: str):
        """Return the batch's pinned fields, or None if it does not exist."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM pilot_batches WHERE factory_id=? AND batch_id=?",
                (factory_id, batch_id),
            ).fetchone()
            return self._assemble_batch(row) if row else None

    def open_batch(self, factory_id: str, batch_id: str, compute_offset):
        """Open a pilot batch, or return the existing one unchanged.

        Everything happens in one IMMEDIATE transaction that starts by
        reading the committed current-version pointer: the batch pins that
        version, that version's correction for the factory and a power
        offset.  The offset is the mean residual of this factory's already
        committed reports when scored with the pinned model (+ pinned
        correction), computed from the running `report_totals` row --
        never by scanning pilot_reports.

        `compute_offset(n, sums, model_dict, correction)` returns the mean
        residual.  If the batch already exists nothing is recomputed or
        written, so a retried open returns the original pins (idempotent).
        Returns (batch, created).
        """
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute("BEGIN IMMEDIATE")
                row = cur.execute(
                    "SELECT * FROM pilot_batches"
                    " WHERE factory_id=? AND batch_id=?",
                    (factory_id, batch_id),
                ).fetchone()
                if row is not None:
                    self._conn.commit()
                    return self._assemble_batch(row), False
                vid = self._current_version_id(cur)
                if vid is None:
                    raise NoCurrentVersion("no model version yet")
                ver = cur.execute(
                    "SELECT model_json FROM versions WHERE id=?", (vid,)
                ).fetchone()
                frow = cur.execute(
                    "SELECT correction FROM factory_stats"
                    " WHERE version_id=? AND factory_id=?",
                    (vid, factory_id),
                ).fetchone()
                if frow is None:
                    raise UnknownFactory(factory_id)
                correction = int(frow["correction"])
                totals = cur.execute(
                    "SELECT n, sums_json FROM report_totals WHERE factory_id=?",
                    (factory_id,),
                ).fetchone()
                if totals is None or totals["n"] <= 0:
                    offset = 0.0
                else:
                    offset = float(compute_offset(
                        totals["n"],
                        _parse_ints(totals["sums_json"]),
                        json.loads(ver["model_json"]),
                        correction,
                    ))
                cur.execute(
                    "INSERT INTO pilot_batches"
                    " (factory_id, batch_id, version_id, correction, offset,"
                    "  opened_at) VALUES (?,?,?,?,?,?)",
                    (
                        factory_id,
                        batch_id,
                        vid,
                        str(correction),
                        offset,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                row = cur.execute(
                    "SELECT * FROM pilot_batches"
                    " WHERE factory_id=? AND batch_id=?",
                    (factory_id, batch_id),
                ).fetchone()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return self._assemble_batch(row), True

    def get_report(self, factory_id: str, batch_id: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM pilot_reports WHERE factory_id=? AND batch_id=?",
                (factory_id, batch_id),
            ).fetchone()
            return self._assemble_report(row) if row else None

    def submit_report(self, factory_id: str, batch_id: str, n, sums, outer,
                      mse_before, mse_after):
        """Insert or replace this batch's summary; update factory totals.

        The old summary (if any) is removed from `report_totals` and the new
        one added in the same transaction, so later opens derive their offset
        from the current summaries only.  Returns {"replaced": bool, "report":
        ...}; raises UnknownBatch if the batch was never opened.  On any error
        the original summary and totals are kept.
        """
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute("BEGIN IMMEDIATE")
                batch = cur.execute(
                    "SELECT 1 FROM pilot_batches WHERE factory_id=? AND batch_id=?",
                    (factory_id, batch_id),
                ).fetchone()
                if batch is None:
                    raise UnknownBatch(f"{factory_id}/{batch_id}")
                old = cur.execute(
                    "SELECT n, sums_json FROM pilot_reports"
                    " WHERE factory_id=? AND batch_id=?",
                    (factory_id, batch_id),
                ).fetchone()
                self._apply_totals(cur, factory_id,
                                   -1 if old else 0,
                                   old["n"] if old else 0,
                                   _parse_ints(old["sums_json"]) if old else None)
                self._apply_totals(cur, factory_id, +1, n, list(sums))
                cur.execute(
                    "INSERT INTO pilot_reports"
                    " (factory_id, batch_id, n, sums_json, outer_json,"
                    "  mse_before, mse_after, reported_at)"
                    " VALUES (?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(factory_id, batch_id) DO UPDATE SET"
                    "  n=excluded.n, sums_json=excluded.sums_json,"
                    "  outer_json=excluded.outer_json,"
                    "  mse_before=excluded.mse_before,"
                    "  mse_after=excluded.mse_after,"
                    "  reported_at=excluded.reported_at",
                    (
                        factory_id, batch_id, int(n), _jints(sums),
                        _jint_matrix(outer), float(mse_before),
                        float(mse_after),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                row = cur.execute(
                    "SELECT * FROM pilot_reports WHERE factory_id=? AND batch_id=?",
                    (factory_id, batch_id),
                ).fetchone()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return {"replaced": old is not None,
                    "report": self._assemble_report(row)}

    @staticmethod
    def _apply_totals(cur, factory_id, sign, n, sums):
        """Add (sign=+1) or subtract (sign=-1) one report from the totals."""
        if sign == 0:
            return
        row = cur.execute(
            "SELECT n, sums_json FROM report_totals WHERE factory_id=?",
            (factory_id,),
        ).fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO report_totals (factory_id, n, sums_json)"
                " VALUES (?,?,?)",
                (factory_id, sign * int(n), _jints([sign * s for s in sums])),
            )
            return
        cur.execute(
            "UPDATE report_totals SET n=?, sums_json=? WHERE factory_id=?",
            (
                row["n"] + sign * int(n),
                _jints([a + sign * b
                        for a, b in zip(_parse_ints(row["sums_json"]), sums)]),
                factory_id,
            ),
        )

    @staticmethod
    def _assemble_batch(row):
        return {
            "factory_id": row["factory_id"],
            "batch_id": row["batch_id"],
            "version": row["version_id"],
            "correction": int(row["correction"]),
            "offset": row["offset"],
            "opened_at": row["opened_at"],
        }

    @staticmethod
    def _assemble_report(row):
        return {
            "factory_id": row["factory_id"],
            "batch_id": row["batch_id"],
            "n": row["n"],
            "sums": _parse_ints(row["sums_json"]),
            "outer": _parse_matrix(row["outer_json"]),
            "mse_before": row["mse_before"],
            "mse_after": row["mse_after"],
            "reported_at": row["reported_at"],
        }

    # ---------------------------------------------------------- helpers
    def _insert_version(self, cur, factories, result) -> int:
        row = cur.execute("SELECT COALESCE(MAX(id), 0) + 1 AS nid FROM versions").fetchone()
        new_id = int(row["nid"])
        cur.execute(
            "INSERT INTO versions (id, lambda, scores_json, model_json, created_at)"
            " VALUES (?,?,?,?,?)",
            (
                new_id,
                result.lam,
                json.dumps({str(k): v for k, v in result.lambda_scores.items()}),
                json.dumps(result.model.to_dict()),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        for f in factories:
            cur.execute(
                "INSERT INTO factory_stats"
                " (version_id, factory_id, n, sums_json, outer_json, correction)"
                " VALUES (?,?,?,?,?,?)",
                (
                    new_id,
                    f["factory_id"],
                    f["n"],
                    _jints(f["sums"]),
                    _jint_matrix(f["outer"]),
                    str(int(f["correction"])),
                ),
            )
        for fid, mse in zip([f["factory_id"] for f in factories],
                            result.factory_errors):
            cur.execute(
                "INSERT INTO factory_errors (version_id, factory_id, mse)"
                " VALUES (?,?,?)",
                (new_id, fid, float(mse)),
            )
        return new_id

    def _set_current(self, cur, version_id: int):
        cur.execute(
            "INSERT INTO meta (key, value) VALUES ('current_version', ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(version_id),),
        )
