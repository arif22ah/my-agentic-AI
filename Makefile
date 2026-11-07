.RECIPEPREFIX = >
.PHONY: sandbox up down logs test lint dev

sandbox:
> docker build -t selfheal-sandbox:latest sandbox/

up: sandbox
> docker compose up -d --build

down:
> docker compose down

logs:
> docker compose logs -f api

dev:
> uvicorn my_agent.api:app --reload

test:
> pytest -q

lint:
> ruff check .
