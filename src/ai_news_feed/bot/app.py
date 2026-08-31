"""python-telegram-bot long-polling adapter for BotWorkerHandlers."""

from __future__ import annotations

import os
from typing import Any

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ai_news_feed.bot.handlers import BotWorkerHandlers, Keyboard
from ai_news_feed.storage.postgres import PostgresRepository


class PythonTelegramBotAPI:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        buttons: Keyboard = (),
    ) -> None:
        markup = None
        if buttons:
            markup = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(button.text, callback_data=button.callback_data)
                        for button in row
                    ]
                    for row in buttons
                ]
            )
        await self._bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=markup,
            disable_web_page_preview=True,
        )


def build_application(
    *,
    token: str,
    repository: PostgresRepository,
    owner_user_id: int,
    interest_profile_id: str = "default",
) -> Application[Any, Any, Any, Any, Any, Any]:
    async def close_repository(
        _application: Application[Any, Any, Any, Any, Any, Any],
    ) -> None:
        await repository.close()

    application = Application.builder().token(token).post_shutdown(close_repository).build()
    handlers = BotWorkerHandlers(
        repository=repository,
        api=PythonTelegramBotAPI(application.bot),
        owner_user_id=owner_user_id,
        interest_profile_id=interest_profile_id,
    )

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if update.effective_chat is not None and update.effective_user is not None:
            await handlers.handle_start(
                chat_id=update.effective_chat.id,
                user_id=update.effective_user.id,
            )

    async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        query = update.callback_query
        if query is None or update.effective_chat is None or update.effective_user is None:
            return
        await query.answer()
        await handlers.handle_callback(
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id,
            data=query.data or "",
        )

    async def text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if (
            update.effective_chat is not None
            and update.effective_user is not None
            and update.effective_message is not None
            and update.effective_message.text is not None
        ):
            await handlers.handle_text(
                chat_id=update.effective_chat.id,
                user_id=update.effective_user.id,
                text=update.effective_message.text,
            )

    application.add_handler(CommandHandler("start", start, filters=filters.ChatType.PRIVATE))
    application.add_handler(CallbackQueryHandler(callback))
    application.add_handler(
        MessageHandler(filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND, text)
    )
    return application


def main() -> None:
    token = _required_env("TELEGRAM_BOT_TOKEN")
    database_url = _required_env("DATABASE_URL")
    owner_user_id = int(_required_env("TELEGRAM_OWNER_USER_ID"))
    repository = PostgresRepository(database_url, pooled=True)
    application = build_application(
        token=token,
        repository=repository,
        owner_user_id=owner_user_id,
        interest_profile_id=os.environ.get("INTEREST_PROFILE_ID", "default"),
    )
    application.run_polling(drop_pending_updates=False, allowed_updates=Update.ALL_TYPES)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value
