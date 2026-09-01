"""Interactive, one-off Telethon session generator -- NOT part of pytest/CI.

Run this yourself, in your own terminal, logged in as the Telegram account
you want the pipeline to read public channels as (the backup account, not
the main one that owns the bot/channel). It will ask for that account's
phone number, then the login code Telegram sends to it, then the 2FA
password too if cloud password is enabled.

Nothing touches disk -- StringSession lives only in memory for this one
run, no .session file is written. The printed value is the TELEGRAM_SESSION
secret; copy it straight into GitHub and don't paste it anywhere else
(it is a live credential, equivalent to being logged in as that account).

Usage:
    uv run python scripts/telethon_login.py <api_id> <api_hash>

<api_id> and <api_hash> come from https://my.telegram.org -> API development
tools, created while logged in as the same backup account.
"""

from __future__ import annotations

import asyncio
import sys

from telethon import TelegramClient
from telethon.sessions import StringSession


async def run(api_id: int, api_hash: str) -> None:
    client = TelegramClient(StringSession(), api_id, api_hash)
    async with client:
        me = await client.get_me()
        print()
        print(f"Logged in as: {me.first_name} (@{me.username}, id={me.id})")
        print("Double-check this is your BACKUP account, not the main one.")
        print()
        print("TELEGRAM_SESSION value (copy everything on the next line):")
        print(client.session.save())


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: uv run python scripts/telethon_login.py <api_id> <api_hash>")
        return 2
    asyncio.run(run(int(sys.argv[1]), sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
