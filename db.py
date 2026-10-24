import sqlite3
import os
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(__file__), "scan_history.db")

def init_db():
    """Initializes the database table for storing scan history."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            is_phishing INTEGER NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            flags TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_scan(url: str, is_phishing: bool, confidence: float, source: str, flags: list):
    """Logs a new URL scan entry into the SQLite database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        flags_str = ", ".join(flags) if isinstance(flags, list) else str(flags)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO scan_logs (url, is_phishing, confidence, source, flags, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (url, 1 if is_phishing else 0, confidence, source, flags_str, timestamp))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database insertion error: {e}")

def get_recent_scans(limit: int = 20):
    """Retrieves recent scan records from the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, url, is_phishing, confidence, source, flags, timestamp
            FROM scan_logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Database query error: {e}")
        return []

def get_scan_stats():
    """Calculates overall scan statistics from the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM scan_logs")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM scan_logs WHERE is_phishing = 1")
        phishing = cursor.fetchone()[0]
        conn.close()
        safe = total - phishing
        return total, phishing, safe
    except Exception as e:
        print(f"Database stats error: {e}")
        return 0, 0, 0
