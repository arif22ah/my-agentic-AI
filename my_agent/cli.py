"""`selfheal "write a function that ..."` : run the loop from the terminal."""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command

from my_agent.agent import build_graph
from my_agent.config import get_settings

log = logging.getLogger(__name__)


def _run(graph, payload: dict, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    cur: dict | Command = payload
    while True:
        for update in graph.stream(cur, config, stream_mode="updates"):
            for node, delta in update.items():
                if node == "__interrupt__":
                    continue
                d = delta if isinstance(delta, dict) else {}
                if node == "plan_task":
                    print("[planner] interface pinned")
                elif node == "write_tests":
                    print("[tester ] wrote test suite")
                elif node == "generate_code":
                    print(f"[coder  ] attempt {d['attempt']} ({d['model_tier']} model)")
                elif node == "run_tests":
                    if d["passed"]:
                        print("[tester ] PASSED (tests + lint)")
                    else:
                        last = d["error_log"].strip().splitlines()[-1:] or [""]
                        print(f"[tester ] FAILED -> looping back | {last[0]}")
        snap = graph.get_state(config)
        if snap.next:  # paused in ask_human
            print("\nBudget spent. Give the coder a hint (empty line = just retry).")
            cur = Command(resume=input("hint> ").strip() or "Try a different approach.")
            continue
        return dict(snap.values)


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="Self-healing code generator")
    p.add_argument("task", help="What the code should do")
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--hitl", action="store_true", help="Ask me for a hint when stuck")
    p.add_argument("--out", default="out/solution.py", help="Where to write the final code")
    p.add_argument("--thread-id", default=None, help="Reuse to resume a checkpointed run")
    args = p.parse_args()

    payload: dict = {"task": args.task, "hitl": args.hitl}
    if args.max_attempts:
        payload["max_attempts"] = args.max_attempts
    thread_id = args.thread_id or str(uuid.uuid4())

    final = None
    url = get_settings().database_url
    if url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            with PostgresSaver.from_conn_string(url) as saver:
                saver.setup()
                final = _run(build_graph(saver), payload, thread_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Postgres unavailable (%s); falling back to InMemorySaver.", exc)

    if final is None:
        from langgraph.checkpoint.memory import InMemorySaver

        final = _run(build_graph(InMemorySaver()), payload, thread_id)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(final.get("code", ""), encoding="utf-8")
    ok = final.get("status") == "passed"
    print(f"\n{'SUCCESS' if ok else 'GAVE UP'} after {final.get('attempt')} attempt(s) "
          f"-> {out} (thread {thread_id})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
