from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional
from collections import deque

import discord
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
OPENAI_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DISCORD_TEST_CHANNEL_ID = os.getenv("DISCORD_TEST_CHANNEL_ID", "")
DISCORD_USE_MESSAGE_CONTENT = os.getenv("DISCORD_USE_MESSAGE_CONTENT", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CHAT_SYSTEM_PROMPT = os.getenv("CHAT_SYSTEM_PROMPT", "").strip()

client = OpenAI()

HISTORY_LIMIT = 6
_channel_histories: dict[int, deque[dict[str, str]]] = {}


def _extract_output_text(response: Any) -> Optional[str]:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    try:
        return response.output[0].content[0].text
    except Exception:
        return None


def _generate_reply_sync(user_text: str, history: list[dict[str, str]]) -> str:
    if CHAT_SYSTEM_PROMPT:
        input_payload = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    else:
        input_payload = []

    input_payload.extend(history)
    input_payload.append({"role": "user", "content": user_text})

    response = client.responses.create(model=OPENAI_MODEL, input=input_payload)
    output_text = _extract_output_text(response)
    return output_text or "Sorry, I could not generate a response right now."


async def _generate_reply(user_text: str, history: list[dict[str, str]]) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _generate_reply_sync, user_text, history)


def _chunk_text(text: str, max_len: int = 1800) -> list[str]:
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


intents = discord.Intents.default()
intents.message_content = DISCORD_USE_MESSAGE_CONTENT

bot = discord.Client(intents=intents)


@bot.event
async def on_ready() -> None:
    print(f"Logged in as {bot.user}")
    if DISCORD_TEST_CHANNEL_ID:
        try:
            channel_id = int(DISCORD_TEST_CHANNEL_ID)
        except ValueError:
            print("[discord] DISCORD_TEST_CHANNEL_ID must be an integer")
            return

        channel = bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await bot.fetch_channel(channel_id)
            except discord.DiscordException as exc:
                print("[discord] Failed to fetch channel", exc)
                return

        try:
            await channel.send("håll shift")
        except discord.DiscordException as exc:
            print("[discord] Failed to send test message", exc)


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author == bot.user:
        return

    if not message.content:
        return

    if not DISCORD_USE_MESSAGE_CONTENT and bot.user not in message.mentions:
        return

    if DISCORD_TEST_CHANNEL_ID:
        try:
            allowed_channel_id = int(DISCORD_TEST_CHANNEL_ID)
        except ValueError:
            print("[discord] DISCORD_TEST_CHANNEL_ID must be an integer")
            return

        if message.channel.id != allowed_channel_id:
            return

    history = _channel_histories.setdefault(
        message.channel.id, deque(maxlen=HISTORY_LIMIT)
    )
    reply_text = await _generate_reply(message.content, list(history))
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": reply_text})
    for chunk in _chunk_text(reply_text):
        await message.channel.send(chunk)


if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise SystemExit("Missing DISCORD_BOT_TOKEN in environment")

    bot.run(DISCORD_BOT_TOKEN)
