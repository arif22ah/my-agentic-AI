"""Best-effort Postgres persistence for runs and attempts.

If DATABASE_URL is unset or the database is unreachable, every function degrades
to a no-op so the agent loop never fails because of bookkeeping.
"""
from __future__ import annotations

import logging
import uuid

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from my_agent.config import get_settings

log = logging.getLogger(__name__)
_pool: ConnectionPool | None = None
_pool_url: str | None = None


def _get_pool() -> ConnectionPool | None:
    global _pool, _pool_url
    url = get_settings().database_url
    if not url:
        return None
    if _pool is None or _pool_url != url:
        try:
            _pool = ConnectionPool(
                url,
                min_size=0,
                max_size=4,
                open=True,
                timeout=2,
                kwargs={"connect_timeout": 2},
            )
            _pool_url = url
            with _pool.connection() as conn:
                conn.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS plan text")
                conn.execute("DELETE FROM runs WHERE status = 'running' AND attempts = 0")
        except Exception:  # noqa: BLE001
            log.warning("Could not initialize Postgres pool; persistence disabled.", exc_info=True)
            return None
    return _pool


def create_run(task: str, run_id: str | None = None) -> str:
    run_id = run_id or str(uuid.uuid4())
    try:
        pool = _get_pool()
        if pool:
            with pool.connection() as conn:
                conn.execute(
                    "INSERT INTO runs (id, task) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    (run_id, task),
                )
    except Exception:  # noqa: BLE001
        log.warning("create_run failed; continuing without persistence", exc_info=True)
    return run_id


def record_attempt(run_id: str, attempt: int, code: str, passed: bool, output: str) -> None:
    try:
        pool = _get_pool()
        if pool:
            with pool.connection() as conn:
                conn.execute(
                    """INSERT INTO attempts (run_id, attempt, code, passed, output)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (run_id, attempt) DO NOTHING""",
                    (run_id, attempt, code, passed, output),
                )
    except Exception:  # noqa: BLE001
        log.warning("record_attempt failed", exc_info=True)


def finish_run(
    run_id: str, status: str, attempts: int, tests: str, final_code: str, plan: str = ""
) -> None:
    try:
        pool = _get_pool()
        if pool:
            with pool.connection() as conn:
                conn.execute(
                    """UPDATE runs SET status=%s, attempts=%s, plan=%s, tests=%s, final_code=%s,
                              finished_at=now() WHERE id=%s""",
                    (status, attempts, plan, tests, final_code, run_id),
                )
    except Exception:  # noqa: BLE001
        log.warning("finish_run failed", exc_info=True)


def list_runs(limit: int = 30) -> list[dict]:
    try:
        pool = _get_pool()
        if not pool:
            return []
        with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id::text AS id, task, status, attempts, created_at, finished_at
                   FROM runs ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            )
            return cur.fetchall()
    except Exception:  # noqa: BLE001
        log.warning("list_runs failed; returning empty history", exc_info=True)
        return []


def get_run(run_id: str) -> dict | None:
    try:
        pool = _get_pool()
        if not pool:
            return None
        with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id::text AS id, task, plan, tests, status, attempts, final_code,
                          created_at, finished_at FROM runs WHERE id=%s""",
                (run_id,),
            )
            run = cur.fetchone()
            if run is None:
                return None
            cur.execute(
                "SELECT attempt, code, passed, output FROM attempts WHERE run_id=%s ORDER BY attempt",
                (run_id,),
            )
            run["attempt_log"] = cur.fetchall()
            return run
    except Exception:  # noqa: BLE001
        log.warning("get_run failed", exc_info=True)
        return None


def clear_runs() -> None:
    try:
        pool = _get_pool()
        if pool:
            with pool.connection() as conn:
                conn.execute("TRUNCATE TABLE attempts, runs CASCADE")
    except Exception:  # noqa: BLE001
        log.warning("clear_runs failed", exc_info=True)
