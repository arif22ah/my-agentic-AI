from __future__ import annotations

import operator
from typing import Annotated, NotRequired, TypedDict


class Attempt(TypedDict):
    """One generate -> test cycle, kept for the audit trail."""

    attempt: int
    code: str
    passed: bool
    output: str


class InputState(TypedDict):
    """What a caller must / may pass in."""

    task: str
    max_attempts: NotRequired[int]
    hitl: NotRequired[bool]  # pause and ask a human for a hint when the budget is spent
    run_id: NotRequired[str]  # lets the API pin the DB id / thread id


class AgentState(InputState, total=False):
    """Full graph state."""

    plan: str  # planner output: summary, exact interface, test focus, pitfalls
    tests: str  # written once by the tester agent, then frozen
    code: str  # latest candidate solution
    error_log: str  # sandbox output of the latest failed run ("" if passing)
    attempt: int
    passed: bool
    status: str  # "passed" | "exhausted"
    signature: str  # fingerprint of the latest failure (detects "same error again")
    stuck_count: int  # consecutive attempts that failed with the same signature
    model_tier: str  # "base" | "strong"
    hint: str  # human guidance, if any
    hitl_used: bool
    history: Annotated[list[Attempt], operator.add]
