"""SQLite persistence for accounts and deployment status."""
import sqlite3
from pathlib import Path


def connect(state: Path):
    state.mkdir(parents=True, mode=0o700, exist_ok=True)
    db = sqlite3.connect(state / 'kdeploy.sqlite3', timeout=30)
    db.execute('PRAGMA foreign_keys = ON')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
            credentials_path TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE TABLE IF NOT EXISTS status (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('not_started', 'running', 'successful', 'failed')),
            acc INTEGER NOT NULL REFERENCES accounts(id), project_path TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE INDEX IF NOT EXISTS status_account ON status(acc, status);
    ''')
    return db


def upsert_account(db, name, path):
    db.execute('''INSERT INTO accounts(name, credentials_path) VALUES (?, ?)
                  ON CONFLICT(name) DO UPDATE SET credentials_path = excluded.credentials_path''', (name, str(path)))
    db.commit()


def set_status(db, run_id, status):
    db.execute("UPDATE status SET status = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE run_id = ? AND status != ?", (status, run_id, status))
    db.commit()


def create_run(db, run_id, profile, project_path):
    db.execute("INSERT INTO status(run_id, status, acc, project_path) SELECT ?, 'not_started', id, ? FROM accounts WHERE name = ?", (run_id, project_path, profile))
    db.commit()
