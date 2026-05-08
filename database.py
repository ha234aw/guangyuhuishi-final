from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "guangyuhuishi.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS current_user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        surgery_stage TEXT,
        surgery_type TEXT,
        main_problem TEXT,
        note TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sensor_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        face_detected INTEGER,
        eye_detected INTEGER,
        brightness REAL,
        stability REAL,
        attention REAL,
        fatigue REAL,
        glare_risk REAL,
        sensor_score REAL,
        advice TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS assessment_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        direction_score REAL,
        contrast_score REAL,
        assessment_score REAL,
        detail_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS training_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        direction_score REAL,
        reading_score REAL,
        search_score REAL,
        training_score REAL,
        completion_rate REAL,
        detail_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ai_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        provider TEXT,
        status TEXT,
        major_issue TEXT,
        training_focus TEXT,
        advice TEXT,
        summary TEXT,
        risk_level TEXT,
        followup TEXT,
        raw_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def clear_all_data() -> None:
    conn = get_conn()
    cur = conn.cursor()
    for table in ["current_user", "sensor_records", "assessment_records", "training_records", "ai_reports"]:
        cur.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()


def upsert_current_user(data: Dict[str, Any]) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM current_user")
    cur.execute(
        """
        INSERT INTO current_user(name, age, surgery_stage, surgery_type, main_problem, note)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("name"),
            data.get("age"),
            data.get("surgery_stage"),
            data.get("surgery_type"),
            data.get("main_problem"),
            data.get("note"),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return int(row_id)


def get_current_user() -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM current_user ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def insert_sensor_record(user_id: Optional[int], result: Dict[str, Any]) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sensor_records(user_id, face_detected, eye_detected, brightness, stability, attention, fatigue, glare_risk, sensor_score, advice)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            int(bool(result.get("face_detected"))),
            int(bool(result.get("eye_detected"))),
            result.get("brightness"),
            result.get("stability"),
            result.get("attention"),
            result.get("fatigue"),
            result.get("glare_risk"),
            result.get("sensor_score"),
            result.get("advice"),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return int(row_id)


def insert_assessment_record(user_id: Optional[int], scores: Dict[str, Any], detail_json: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO assessment_records(user_id, direction_score, contrast_score, assessment_score, detail_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            scores.get("direction_score"),
            scores.get("contrast_score"),
            scores.get("assessment_score"),
            detail_json,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return int(row_id)


def insert_training_record(user_id: Optional[int], scores: Dict[str, Any], detail_json: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO training_records(user_id, direction_score, reading_score, search_score, training_score, completion_rate, detail_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            scores.get("direction_score"),
            scores.get("reading_score"),
            scores.get("search_score"),
            scores.get("training_score"),
            scores.get("completion_rate"),
            detail_json,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return int(row_id)


def insert_ai_report(user_id: Optional[int], report: Dict[str, Any]) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ai_reports(user_id, provider, status, major_issue, training_focus, advice, summary, risk_level, followup, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            report.get("provider"),
            report.get("status"),
            report.get("major_issue"),
            report.get("training_focus"),
            report.get("advice"),
            report.get("summary"),
            report.get("risk_level"),
            report.get("followup"),
            json.dumps(report, ensure_ascii=False, indent=2),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return int(row_id)


def fetch_latest(table_name: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def fetch_all(table_name: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name} ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _avg(table_name: str, field: str) -> float:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT AVG({field}) as v FROM {table_name}")
    row = cur.fetchone()
    conn.close()
    return round(float(row["v"] or 0.0), 1)


def build_summary_stats() -> Dict[str, Any]:
    latest_report = fetch_latest("ai_reports")
    return {
        "assessment_count": len(fetch_all("assessment_records")),
        "training_count": len(fetch_all("training_records")),
        "sensor_count": len(fetch_all("sensor_records")),
        "avg_assessment": _avg("assessment_records", "assessment_score"),
        "avg_training": _avg("training_records", "training_score"),
        "avg_sensor": _avg("sensor_records", "sensor_score"),
        "avg_completion": _avg("training_records", "completion_rate"),
        "latest_risk": latest_report["risk_level"] if latest_report else "暂无",
    }
