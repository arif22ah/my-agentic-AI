"""Offline tests: a scripted fake LLM drives the real graph and the real (local) sandbox."""
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from my_agent.config import get_settings
from my_agent.utils import nodes

TESTS = "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
BAD = "def add(a, b):\n    return a - b\n"
GOOD = "def add(a, b):\n    return a + b\n"
LINT_BAD = "def add(a, b):\n    return a + b + undefined_name * 0\n"


@pytest.fixture(autouse=True)
def local_env(monkeypatch):
    monkeypatch.setenv("SANDBOX_MODE", "local")
    monkeypatch.setenv("DATABASE_URL", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def scripted_llm(code_versions, seen=None):
    calls = {"n": 0}

    def fake(schema, messages, tier="base"):
        if schema is nodes.Plan:
            return nodes.Plan(summary="s", interface="add(a, b)", test_focus=[], pitfalls=[])
        if schema is nodes.GeneratedTests:
            return nodes.GeneratedTests(tests=TESTS)
        if seen is not None:
            seen.append((tier, messages[-1].content))
        i = min(calls["n"], len(code_versions) - 1)
        calls["n"] += 1
        return nodes.CodeSolution(analysis="fake", code=code_versions[i])

    return fake


def build(saver=None):
    from my_agent.agent import build_graph

    return build_graph(saver)


def test_heals_after_one_failure(monkeypatch):
    monkeypatch.setattr(nodes, "_llm_call", scripted_llm([BAD, GOOD]))
    out = build().invoke({"task": "add two numbers", "max_attempts": 4})
    assert out["status"] == "passed" and out["attempt"] == 2
    assert [h["passed"] for h in out["history"]] == [False, True]


def test_gives_up_when_budget_spent(monkeypatch):
    monkeypatch.setattr(nodes, "_llm_call", scripted_llm([BAD]))
    out = build().invoke({"task": "add two numbers", "max_attempts": 2})
    assert out["status"] == "exhausted" and out["attempt"] == 2


def test_error_log_and_plan_are_fed_back(monkeypatch):
    seen = []
    monkeypatch.setattr(nodes, "_llm_call", scripted_llm([BAD, GOOD], seen))
    build().invoke({"task": "add", "max_attempts": 3})
    assert "Failure log" not in seen[0][1] and "add(a, b)" in seen[0][1]  # plan present
    assert "Failure log" in seen[1][1] and "def add(a, b)" in seen[1][1]


def test_escalates_to_strong_model_when_stuck(monkeypatch):
    seen = []
    monkeypatch.setattr(nodes, "_llm_call", scripted_llm([BAD], seen))
    build().invoke({"task": "add", "max_attempts": 3})
    tiers = [t for t, _ in seen]
    assert tiers[0] == "base" and tiers[-1] == "strong"


def test_lint_failure_triggers_retry(monkeypatch):
    pytest.importorskip("ruff")
    monkeypatch.setattr(nodes, "_llm_call", scripted_llm([LINT_BAD, GOOD]))
    out = build().invoke({"task": "add", "max_attempts": 3})
    assert out["status"] == "passed" and out["attempt"] == 2
    assert "undefined" in out["history"][0]["output"].lower()


def test_human_hint_resumes_the_run(monkeypatch):
    seen = []
    monkeypatch.setattr(nodes, "_llm_call", scripted_llm([BAD, GOOD], seen))
    g = build(InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    g.invoke({"task": "add", "max_attempts": 1, "hitl": True}, cfg)
    assert g.get_state(cfg).next == ("ask_human",)
    out = g.invoke(Command(resume="use addition"), cfg)
    assert out["status"] == "passed"
    assert "use addition" in seen[-1][1]


def test_api_roundtrip(monkeypatch):
    from fastapi.testclient import TestClient

    from my_agent.api import app

    monkeypatch.setattr(nodes, "_llm_call", scripted_llm([BAD, GOOD]))
    with TestClient(app) as c:
        rid = c.post("/api/runs", json={"task": "add two numbers", "max_attempts": 3}).json()["run_id"]
        body = c.get(f"/api/runs/{rid}/events").text
    assert "event: run_tests" in body and "event: done" in body and '"status": "passed"' in body
