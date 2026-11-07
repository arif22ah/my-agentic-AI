from __future__ import annotations

from functools import cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = "mock-key"
    llm_model: str = "openai:gpt-4o-mini"
    strong_model: str | None = None  # used when the loop is stuck (e.g. "openai:o4-mini")
    escalate_at_attempt: int = 3  # switch to strong_model from this attempt on
    max_attempts: int = 5
    lint: bool = True  # ruff (E9,F,B) must pass after the tests do
    max_parallel_sandboxes: int = 2

    sandbox_mode: Literal["docker", "local"] = "docker"
    sandbox_image: str = "selfheal-sandbox:latest"
    sandbox_timeout_s: int = 60
    sandbox_mem: str = "1g"
    sandbox_cpus: float = 1.0

    database_url: str | None = None
    max_log_chars: int = 6000


@cache
def get_settings() -> Settings:
    return Settings()
