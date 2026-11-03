"""
Module 4A: Persistence & Regulatory Compliance Audit Store
Stores timestamped decision events, input payloads, causal explanations, and recourse plans.
"""

import os
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_trail.db")


def init_db(db_path: str = DB_PATH):
    """Initializes the SQLite compliance audit table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS decision_audits (
        request_id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        raw_features TEXT NOT NULL,
        prediction INTEGER NOT NULL,
        probability REAL NOT NULL,
        causal_attributions TEXT,
        recourse_pathways TEXT,
        latency_ms REAL
    )
    """)
    conn.commit()
    conn.close()


class AuditStore:
    """Manages recording and retrieval of decision audit events."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._shared_conn = None
        if ":memory:" in db_path:
            self._shared_conn = sqlite3.connect(db_path)
            cursor = self._shared_conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_audits (
                request_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                raw_features TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                causal_attributions TEXT,
                recourse_pathways TEXT,
                latency_ms REAL
            )
            """)
            self._shared_conn.commit()
        else:
            init_db(self.db_path)

    def _get_connection(self):
        if self._shared_conn is not None:
            return self._shared_conn
        return sqlite3.connect(self.db_path)

    def log_decision(
        self,
        raw_features: Dict[str, Any],
        prediction: int,
        probability: float,
        causal_attributions: Optional[List[Dict[str, Any]]] = None,
        recourse_pathways: Optional[List[Dict[str, Any]]] = None,
        latency_ms: float = 0.0
    ) -> str:
        """Persists a single inference and explainability decision event."""
        request_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO decision_audits (
            request_id, timestamp, raw_features, prediction,
            probability, causal_attributions, recourse_pathways, latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request_id,
            ts,
            json.dumps(raw_features),
            int(prediction),
            float(probability),
            json.dumps(causal_attributions or []),
            json.dumps(recourse_pathways or []),
            float(latency_ms)
        ))
        conn.commit()
        if self._shared_conn is None:
            conn.close()
        return request_id

    def get_recent_audits(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves recent decision audit events for regulatory review."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM decision_audits ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()

        records = []
        for r in rows:
            records.append({
                "request_id": r["request_id"],
                "timestamp": r["timestamp"],
                "prediction": "Churn" if r["prediction"] == 1 else "No Churn",
                "probability": round(r["probability"], 4),
                "raw_features": json.loads(r["raw_features"]),
                "causal_attributions": json.loads(r["causal_attributions"] or "[]"),
                "recourse_pathways": json.loads(r["recourse_pathways"] or "[]"),
                "latency_ms": r["latency_ms"]
            })
        if self._shared_conn is None:
            conn.close()
        return records
