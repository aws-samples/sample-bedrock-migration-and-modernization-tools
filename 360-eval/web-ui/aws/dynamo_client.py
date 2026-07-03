"""OFFLINE metadata store — SQLite replacement for DynamoDB.

Keeps the exact public surface of the cloud dynamo_client (same function names,
signatures and returned item shapes) so app.py is unchanged.

Each logical table is a generic document store:
    (user_id TEXT, sort_key TEXT, data TEXT-JSON, PRIMARY KEY(user_id, sort_key))
The full item (minus the two key columns) is stored as JSON in `data`; reads
return {user_id, <sort_key>: val, **json.loads(data)} — the same dict shape the
DynamoDB layer returned. No Decimal types are involved (app._decimal_to_native
still works on plain ints/floats).

Concurrency: the Flask request threads, the queue-poller thread, and each
_launch_local_eval worker thread all touch the DB in one process. We use WAL +
busy_timeout, a module-level RLock to serialize writers within the process, and a
short-lived connection per call.
"""
import os
import json
import sqlite3
import threading
from datetime import datetime, timezone

from aws._local import DB_PATH, ensure_data_dir

# Logical table names (kept as exported constants; now used as SQLite table names).
EVALUATIONS_TABLE = os.environ.get('EVALUATIONS_TABLE', 'evaluations')
REPORTS_TABLE = os.environ.get('REPORTS_TABLE', 'reports')
CREDENTIALS_TABLE = os.environ.get('CREDENTIALS_TABLE', 'credentials')
MAX_REPORTS_PER_USER = int(os.environ.get('MAX_REPORTS_PER_USER', '10'))

# Map each logical table to (physical_table_name, sort_key_column).
# Physical names avoid the '360eval-' prefix so they're valid SQL identifiers.
_TABLES = {
    EVALUATIONS_TABLE: ('evaluations', 'eval_id'),
    REPORTS_TABLE: ('reports', 'report_id'),
    CREDENTIALS_TABLE: ('credentials', 'provider'),
}

_LOCK = threading.RLock()
_initialized = False


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute('PRAGMA busy_timeout=30000;')
    return conn


def _init_db():
    global _initialized
    if _initialized:
        return
    with _LOCK:
        if _initialized:
            return
        ensure_data_dir()
        conn = _connect()
        try:
            conn.execute('PRAGMA journal_mode=WAL;')
            conn.execute('PRAGMA synchronous=NORMAL;')
            for physical, _sk in {v[0]: v[1] for v in _TABLES.values()}.items():
                conn.execute(
                    f'CREATE TABLE IF NOT EXISTS "{physical}" ('
                    'user_id TEXT NOT NULL, sort_key TEXT NOT NULL, data TEXT NOT NULL, '
                    'PRIMARY KEY (user_id, sort_key))'
                )
            conn.commit()
        finally:
            conn.close()
        _initialized = True


def _resolve(logical_table):
    return _TABLES[logical_table]


def _row_to_item(physical, sort_col, user_id, sort_val, data_json):
    item = json.loads(data_json)
    item['user_id'] = user_id
    item[sort_col] = sort_val
    return item


def _read_one(logical_table, user_id, sort_val):
    _init_db()
    physical, sort_col = _resolve(logical_table)
    with _LOCK:
        conn = _connect()
        try:
            cur = conn.execute(
                f'SELECT data FROM "{physical}" WHERE user_id=? AND sort_key=?',
                (user_id, sort_val),
            )
            row = cur.fetchone()
        finally:
            conn.close()
    if not row:
        return None
    return _row_to_item(physical, sort_col, user_id, sort_val, row[0])


def _read_many(logical_table, user_id):
    _init_db()
    physical, sort_col = _resolve(logical_table)
    with _LOCK:
        conn = _connect()
        try:
            cur = conn.execute(
                f'SELECT sort_key, data FROM "{physical}" WHERE user_id=?',
                (user_id,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
    return [_row_to_item(physical, sort_col, user_id, r[0], r[1]) for r in rows]


def _write(logical_table, user_id, sort_val, item):
    """INSERT OR REPLACE the full item (key columns stripped from the JSON blob)."""
    _init_db()
    physical, sort_col = _resolve(logical_table)
    blob = {k: v for k, v in item.items() if k not in ('user_id', sort_col)}
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                f'INSERT OR REPLACE INTO "{physical}" (user_id, sort_key, data) VALUES (?,?,?)',
                (user_id, sort_val, json.dumps(blob)),
            )
            conn.commit()
        finally:
            conn.close()


def _patch(logical_table, user_id, sort_val, updates):
    """Read-modify-write a subset of fields (under the lock)."""
    _init_db()
    physical, sort_col = _resolve(logical_table)
    with _LOCK:
        conn = _connect()
        try:
            cur = conn.execute(
                f'SELECT data FROM "{physical}" WHERE user_id=? AND sort_key=?',
                (user_id, sort_val),
            )
            row = cur.fetchone()
            if not row:
                return
            data = json.loads(row[0])
            for k, v in updates.items():
                if k in ('user_id', sort_col):
                    continue
                data[k] = v
            conn.execute(
                f'UPDATE "{physical}" SET data=? WHERE user_id=? AND sort_key=?',
                (json.dumps(data), user_id, sort_val),
            )
            conn.commit()
        finally:
            conn.close()


def _delete(logical_table, user_id, sort_val):
    _init_db()
    physical, _sort_col = _resolve(logical_table)
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                f'DELETE FROM "{physical}" WHERE user_id=? AND sort_key=?',
                (user_id, sort_val),
            )
            conn.commit()
        finally:
            conn.close()


def _scan_all(logical_table):
    _init_db()
    physical, sort_col = _resolve(logical_table)
    with _LOCK:
        conn = _connect()
        try:
            cur = conn.execute(f'SELECT user_id, sort_key, data FROM "{physical}"')
            rows = cur.fetchall()
        finally:
            conn.close()
    return [_row_to_item(physical, sort_col, r[0], r[1], r[2]) for r in rows]


# --- Evaluations ---

def get_evaluations(user_id):
    return _read_many(EVALUATIONS_TABLE, user_id)


def get_evaluation(user_id, eval_id):
    return _read_one(EVALUATIONS_TABLE, user_id, eval_id)


def put_evaluation(user_id, eval_id, eval_config):
    now = _now_iso()
    item = {
        'user_id': user_id,
        'eval_id': eval_id,
        'created_at': now,
        'updated_at': now,
        'status': 'configuring',
        'progress': 0,
        **eval_config,
    }
    _write(EVALUATIONS_TABLE, user_id, eval_id, item)
    return item


def update_evaluation(user_id, eval_id, **updates):
    updates['updated_at'] = _now_iso()
    _patch(EVALUATIONS_TABLE, user_id, eval_id, updates)


def delete_evaluation(user_id, eval_id):
    _delete(EVALUATIONS_TABLE, user_id, eval_id)


def scan_evaluations_by_status(statuses):
    """Return all evaluations (across users) whose status is one of `statuses`."""
    wanted = set(statuses)
    return [it for it in _scan_all(EVALUATIONS_TABLE) if it.get('status') in wanted]


def claim_eval_for_launch(user_id, eval_id):
    """Atomically claim a queued eval for launch.

    Sets ecs_task_arn='PENDING' WHERE status='queued' AND ecs_task_arn is absent/null.
    Returns True if the claim succeeded, False otherwise. Single guarded UPDATE
    under the lock so concurrent pollers can't both win.
    """
    _init_db()
    physical, _sk = _resolve(EVALUATIONS_TABLE)
    now = _now_iso()
    with _LOCK:
        conn = _connect()
        try:
            cur = conn.execute(
                f'UPDATE "{physical}" '
                "SET data=json_set(json_set(data,'$.ecs_task_arn','PENDING'),'$.updated_at',?) "
                'WHERE user_id=? AND sort_key=? '
                "AND json_extract(data,'$.status')='queued' "
                "AND json_extract(data,'$.ecs_task_arn') IS NULL",
                (now, user_id, eval_id),
            )
            conn.commit()
            return cur.rowcount == 1
        finally:
            conn.close()


def scan_all_evaluations():
    """All evaluations across all users (admin dashboard)."""
    return _scan_all(EVALUATIONS_TABLE)


# --- Reports ---

def get_reports(user_id):
    return _read_many(REPORTS_TABLE, user_id)


def count_reports(user_id):
    return len(_read_many(REPORTS_TABLE, user_id))


def put_report(user_id, report_id, report_config):
    if count_reports(user_id) >= MAX_REPORTS_PER_USER:
        raise ValueError(
            f'Report limit reached ({MAX_REPORTS_PER_USER}). '
            'Delete existing reports before generating new ones.'
        )
    item = {
        'user_id': user_id,
        'report_id': report_id,
        'created_at': _now_iso(),
        **report_config,
    }
    _write(REPORTS_TABLE, user_id, report_id, item)
    return item


def update_report(user_id, report_id, **updates):
    updates['updated_at'] = _now_iso()
    _patch(REPORTS_TABLE, user_id, report_id, updates)


def delete_report(user_id, report_id):
    _delete(REPORTS_TABLE, user_id, report_id)


def scan_all_reports():
    """All reports across all users (admin dashboard)."""
    return _scan_all(REPORTS_TABLE)


# --- User Credentials ---

def get_credentials(user_id):
    items = _read_many(CREDENTIALS_TABLE, user_id)
    return [
        {'provider': it['provider'], 'key_alias': it.get('key_alias', ''),
         'updated_at': it.get('updated_at', '')}
        for it in items
    ]


def get_credential_encrypted(user_id, provider):
    item = _read_one(CREDENTIALS_TABLE, user_id, provider)
    if item:
        return item.get('encrypted_key')
    return None


def put_credential(user_id, provider, encrypted_key, key_alias):
    item = {
        'user_id': user_id,
        'provider': provider,
        'encrypted_key': encrypted_key,
        'key_alias': key_alias,
        'updated_at': _now_iso(),
    }
    _write(CREDENTIALS_TABLE, user_id, provider, item)


def delete_credential(user_id, provider):
    _delete(CREDENTIALS_TABLE, user_id, provider)
