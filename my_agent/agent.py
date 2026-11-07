"""Compiles the nodes and edges into the LangGraph.

  START -> init_run -> plan_task -> write_tests -> generate_code -> run_tests
                                                        ^               |
                                      failed, budget left|---------------+
                                                        |               | failed, budget spent
                                      ask_human (hint) -+<--------------+  (only if hitl=True)
                                                                        | passed / give up
                                                                        v
                                                                     finalize -> END
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from my_agent.utils import nodes
from my_agent.utils.state import AgentState, InputState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    b = StateGraph(AgentState, input_schema=InputState)

    b.add_node("init_run", nodes.init_run)
    b.add_node("plan_task", nodes.plan_task)
    b.add_node("write_tests", nodes.write_tests)
    b.add_node("generate_code", nodes.generate_code)
    b.add_node("run_tests", nodes.run_tests)
    b.add_node("ask_human", nodes.ask_human, destinations=("generate_code",))
    b.add_node("finalize", nodes.finalize)

    b.add_edge(START, "init_run")
    b.add_edge("init_run", "plan_task")
    b.add_edge("plan_task", "write_tests")
    b.add_edge("write_tests", "generate_code")
    b.add_edge("generate_code", "run_tests")
    b.add_conditional_edges(
        "run_tests",
        nodes.route_after_tests,
        {"generate_code": "generate_code", "ask_human": "ask_human", "finalize": "finalize"},
    )
    b.add_edge("finalize", END)
    return b.compile(checkpointer=checkpointer)


# Exposed for LangGraph Studio / `langgraph dev` (they provide their own persistence).
graph = build_graph()
