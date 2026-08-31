# Deployment hand-off

These files are templates only. No deployment is performed from the repository.

## 1. Prepare the dedicated database

Create a separate Neon project for AI News Feed (Postgres 18), with its own database,
role and TLS-only `DATABASE_URL`. Do not reuse any project, database, role or schema from
`rus-ai.com`. From a trusted machine with the new URL exported, apply migrations once:

```bash
uv sync --frozen
uv run alembic upgrade head
```

Do not run migrations automatically from `bot-worker` or `pipeline-runner`.

## 2. Prepare the isolated VPS service later

When VPS access is available, create a non-login, non-sudo `ai-news-feed` OS user, the
directories `/opt/ai-news-feed/{current,venv}` and `/etc/ai-news-feed`, and a dedicated
Python 3.12 venv. Install the locked project into that venv. Keep the application checkout
read-only for the service user.

Copy `deploy/.env.example` to `/etc/ai-news-feed/bot.env`, fill only the new project's
secrets, set owner/mode to `root:root 0600`, and copy the unit to
`/etc/systemd/system/ai-news-feed-bot.service`. Then an operator can run daemon-reload,
enable/start the unit and inspect it with `systemctl status`/`journalctl`. Do not edit or
restart the Streamlit/Nginx unit, venv or files of `rus-ai.com`; long polling needs no
Nginx route or inbound port.

The bot creates the initial `default` InterestProfile when the owner first uses
“Редактировать интересы”. For one bot token, run exactly one bot-worker replica and keep
Telegram webhook mode disabled.

## 3. Configure GitHub Actions later

Add repository secrets `DATABASE_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHANNEL_ID`,
`OPENAI_API_KEY`, `OPENAI_SCREENING_MODEL` and `OPENAI_SUMMARY_MODEL`. If any enabled
source uses `collector=telethon`, also add `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` and
`TELEGRAM_SESSION`. The workflow intentionally runs `pipeline-runner` on a GitHub-hosted
runner, never on the VPS.

LLM model choice is explicit rather than hidden in code:

1. One small model for both steps is cheapest and simplest, but summary quality can suffer.
2. A small screening model plus a stronger summary model is the recommended starting
   point: screening dominates call count, while only accepted clusters pay for quality.
3. One stronger model for both steps is easiest to evaluate, but has the highest recurring
   cost. Choose exact model IDs only after checking current access and pricing.

PostgreSQL integration testing also has three viable shapes. This repository chooses a
Postgres 18 GitHub Actions service container: it validates real constraints/transactions,
is reproducible and needs no developer Docker, at the cost of CI startup time. Local
Testcontainers offers closer developer feedback but requires Docker. Mocking SQLAlchemy is
fastest, but cannot validate PostgreSQL enums, JSONB, constraints or transaction semantics,
so it is kept only for unit-level fakes and not used as the database acceptance test.
