"""FastAPI server: start runs, stream live progress (SSE), browse history, send hints.

    uvicorn my_agent.api:app --reload      ->  http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_agent.agent import build_graph
from my_agent.config import get_settings
from my_agent.utils import db, nodes

# Psycopg 3 async requires WindowsSelectorEventLoopPolicy on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()
log = logging.getLogger(__name__)
TERMINAL = {"done", "error"}


class EventHub:
    """In-memory, replayable per-run event log with async fan-out (powers the SSE stream)."""

    def __init__(self) -> None:
        self._events: dict[str, list[dict]] = {}
        self._conds: dict[str, asyncio.Condition] = {}

    def open(self, run_id: str) -> None:
        self._events[run_id] = []
        self._conds[run_id] = asyncio.Condition()

    def exists(self, run_id: str) -> bool:
        return run_id in self._events

    async def emit(self, run_id: str, type_: str, data: dict) -> None:
        async with self._conds[run_id]:
            evs = self._events[run_id]
            evs.append({"id": len(evs), "type": type_, "data": data})
            self._conds[run_id].notify_all()

    async def stream(self, run_id: str, start: int = 0):
        cond, i = self._conds[run_id], start
        while True:
            evs = self._events[run_id]
            while i < len(evs):
                ev = evs[i]
                i += 1
                yield ev
                if ev["type"] in TERMINAL:
                    return
            async with cond:
                if i >= len(self._events[run_id]):
                    try:
                        await asyncio.wait_for(cond.wait(), 15)
                    except asyncio.TimeoutError:
                        yield None  # keep-alive tick


hub = EventHub()
_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    t = asyncio.create_task(coro)
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


@asynccontextmanager
async def lifespan(app: FastAPI):
    url = get_settings().database_url
    if url:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            async with AsyncPostgresSaver.from_conn_string(url) as saver:
                await saver.setup()
                app.state.graph = build_graph(saver)
                yield
                return
        except Exception as exc:  # noqa: BLE001
            log.warning("Postgres unavailable (%s); falling back to InMemorySaver.", exc)

    from langgraph.checkpoint.memory import InMemorySaver

    app.state.graph = build_graph(InMemorySaver())
    yield


app = FastAPI(title="Self-Healing Code Generator", lifespan=lifespan)


class RunRequest(BaseModel):
    task: str = Field(min_length=5, max_length=4000)
    max_attempts: int = Field(5, ge=1, le=15)
    hitl: bool = False


class HintRequest(BaseModel):
    hint: str = Field(min_length=1, max_length=2000)


def _summarize(node: str, delta: object) -> dict:
    if not isinstance(delta, dict):
        return {}
    out = {k: v for k, v in delta.items() if isinstance(v, (str, int, bool))}
    if node == "run_tests" and delta.get("history"):
        out["record"] = delta["history"][0]
    return out


async def _drive(graph, run_id: str, payload) -> None:
    config = {"configurable": {"thread_id": run_id}}
    try:
        async for update in graph.astream(payload, config, stream_mode="updates"):
            for node, delta in update.items():
                if node == "__interrupt__":
                    value = delta[0].value if delta else {}
                    await hub.emit(run_id, "needs_hint", value)
                else:
                    await hub.emit(run_id, node, _summarize(node, delta))
        snap = await graph.aget_state(config)
        if not snap.next:
            v = snap.values
            await hub.emit(run_id, "done", {"status": v.get("status"),
                                            "attempt": v.get("attempt"), "code": v.get("code", "")})
    except Exception as e:  # noqa: BLE001
        await asyncio.to_thread(db.finish_run, run_id, "error", 0, "", "", "")
        await hub.emit(run_id, "error", {"message": f"{type(e).__name__}: {e}"})


@app.get("/healthz")
async def healthz() -> dict:
    s = get_settings()
    return {"ok": True, "mode": "offline_demo" if nodes._is_mock_api_key() else "live_llm", "model": s.llm_model}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.post("/api/runs", status_code=202)
async def start_run(req: RunRequest, request: Request) -> dict:
    run_id = str(uuid.uuid4())
    hub.open(run_id)
    payload = {"task": req.task, "max_attempts": req.max_attempts, "hitl": req.hitl,
               "run_id": run_id}
    _spawn(_drive(request.app.state.graph, run_id, payload))
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}/events")
async def events(run_id: str, request: Request) -> StreamingResponse:
    if not hub.exists(run_id):
        raise HTTPException(404, "Unknown run (live events are kept in memory); use GET /api/runs/{id}.")
    start = int(request.headers.get("last-event-id", "-1")) + 1

    async def gen():
        async for ev in hub.stream(run_id, start):
            if ev is None:
                yield ": keepalive\n\n"
            else:
                yield f"id: {ev['id']}\nevent: {ev['type']}\ndata: {json.dumps(ev['data'])}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/runs/{run_id}/hint", status_code=202)
async def give_hint(run_id: str, body: HintRequest, request: Request) -> dict:
    if not hub.exists(run_id):
        raise HTTPException(404, "Unknown run.")
    graph = request.app.state.graph
    snap = await graph.aget_state({"configurable": {"thread_id": run_id}})
    if not snap.next:
        raise HTTPException(409, "Run is not waiting for a hint.")
    _spawn(_drive(graph, run_id, Command(resume=body.hint)))
    return {"resumed": True}


@app.get("/api/runs")
async def list_runs() -> list[dict]:
    return await asyncio.to_thread(db.list_runs)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    run = await asyncio.to_thread(db.get_run, run_id)
    if run is None:
        raise HTTPException(404, "Run not found (is DATABASE_URL set?).")
    return run


@app.delete("/api/runs")
async def clear_runs_api() -> dict:
    await asyncio.to_thread(db.clear_runs)
    return {"cleared": True}


class ModeRequest(BaseModel):
    offline: bool


@app.post("/api/mode")
async def toggle_mode(body: ModeRequest) -> dict:
    nodes.set_force_offline(body.offline)
    s = get_settings()
    return {"mode": "offline_demo" if nodes._is_mock_api_key() else "live_llm", "model": s.llm_model}
