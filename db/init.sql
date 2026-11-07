CREATE TABLE IF NOT EXISTS runs (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    task        text        NOT NULL,
    plan        text,
    tests       text,
    status      text        NOT NULL DEFAULT 'running',
    attempts    integer     NOT NULL DEFAULT 0,
    final_code  text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

ALTER TABLE runs ADD COLUMN IF NOT EXISTS plan text;

CREATE TABLE IF NOT EXISTS attempts (
    id         bigserial PRIMARY KEY,
    run_id     uuid        NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    attempt    integer     NOT NULL,
    code       text        NOT NULL,
    passed     boolean     NOT NULL,
    output     text        NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, attempt)
);

CREATE INDEX IF NOT EXISTS attempts_run_idx ON attempts (run_id);
