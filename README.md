# Self-Healing Code Generator

Describe code in plain English. A team of LangGraph agents plans it, writes tests, writes the
code, runs everything in a locked-down Docker sandbox, and feeds failures back to the coder
until the tests and lint pass. Watch it happen live in a web UI, or drive it from the CLI/API.

```
START -> init_run -> plan_task -> write_tests -> generate_code -> run_tests -> finalize -> END
                                                      ^               |
                                    failed, budget left|---------------+
                                                      |               | failed, budget spent
                                    ask_human (hint) -+<--------------+  (if hitl enabled)
```

## What makes it "advanced"

| Feature | How |
|---|---|
| **Planner agent** | Pins the exact interface so tester and coder can't disagree on signatures |
| **Frozen tests** | Written once, then immutable, so the coder can't weaken them to pass |
| **Quality gate** | pytest first, then ruff (`E9,F,B`: real bugs only) inside the sandbox |
| **Stuck detection** | Failures are fingerprinted; a repeat means "change approach" |
| **Model escalation** | Switches to `STRONG_MODEL` when stuck or from attempt `ESCALATE_AT_ATTEMPT` |
| **Human-in-the-loop** | `interrupt()` pauses the run when the budget is spent; a hint resumes it (+3 attempts) |
| **Live UI + API** | FastAPI, SSE event stream with replay (`Last-Event-ID`), run history, hints |
| **Durable** | `AsyncPostgresSaver` checkpoints + readable `runs`/`attempts` tables |
| **Safe sandbox** | No network, CPU/memory/PID caps, caps dropped, `nobody`, timeouts, bounded concurrency |
| **PyTorch-ready** | CPU torch + numpy live in the sandbox image, so generated ML code really runs |

## Run it

```bash
cp .env.example .env          # add OPENAI_API_KEY (optionally STRONG_MODEL)
make up                       # builds sandbox image, starts Postgres + API + UI
# open http://localhost:8000
```

Terminal instead:

```bash
docker compose run --rm cli selfheal "Write a numerically stable softmax in PyTorch" --hitl
```

Local dev (no containers for the app):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && cp .env.example .env
make sandbox && docker compose up -d db
make dev                      # http://localhost:8000
langgraph dev                 # LangGraph Studio
make test                     # offline: scripted fake LLM, no API key
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/runs` | `{task, max_attempts?, hitl?}` -> `{run_id}` |
| GET | `/api/runs/{id}/events` | SSE: `plan_task`, `write_tests`, `generate_code`, `run_tests`, `needs_hint`, `done`, `error` |
| POST | `/api/runs/{id}/hint` | `{hint}` resumes a paused run |
| GET | `/api/runs`, `/api/runs/{id}` | History and per-attempt code/logs (needs Postgres) |

## Configuration (`.env`)

`LLM_MODEL`, `STRONG_MODEL`, `ESCALATE_AT_ATTEMPT`, `MAX_ATTEMPTS`, `LINT`,
`SANDBOX_MODE` (`docker`|`local`), `SANDBOX_TIMEOUT_S`, `SANDBOX_MEM`, `SANDBOX_CPUS`,
`MAX_PARALLEL_SANDBOXES`, `DATABASE_URL`.

## Security notes

- Mounting `/var/run/docker.sock` lets the API container control the host's Docker daemon. Fine
  for a dev box; in production use rootless Docker, gVisor/Firecracker or a remote sandbox.
- The API has no authentication. Put it behind a reverse proxy or add an API-key dependency
  before exposing it, since anyone who can reach it can spend your LLM budget.
- LLM-written tests can be wrong: green means "meets the spec the tester wrote". Review `runs.tests`.
- Never use `SANDBOX_MODE=local` with untrusted prompts on a machine you care about.
- Live SSE events are in memory (single API process). Scale-out needs Redis/Postgres pub-sub.

## Ideas to extend

Token/cost accounting per run, mypy as a second gate, multi-file projects, other languages
(swap the sandbox image and test command), an LLM-as-judge for test quality.
