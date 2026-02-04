from __future__ import annotations

import asyncio
import os
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from collections import deque

import discord
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "discord_bot"


def _load_prompt_file(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
OPENAI_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DISCORD_TEST_CHANNEL_ID = os.getenv("DISCORD_TEST_CHANNEL_ID", "")
DISCORD_USE_MESSAGE_CONTENT = os.getenv("DISCORD_USE_MESSAGE_CONTENT", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CHAT_SYSTEM_PROMPT = _load_prompt_file("system_prompt.txt").strip()

START_AGENT_SYSTEM_PROMPT = _load_prompt_file("start_agent_system.txt").strip()
if not START_AGENT_SYSTEM_PROMPT:
    START_AGENT_SYSTEM_PROMPT = (
        "When the user confirms they want to start the agent, respond with a hidden command:\n"
        "<START_AGENT>{\"prompt\": \"...\"}</START_AGENT>\n"
        "Do not include any other text inside the START_AGENT block. You may include normal user-visible text outside it."
    )

client = OpenAI()

HISTORY_LIMIT = 6
_channel_histories: dict[int, deque[dict[str, str]]] = {}
_channel_tail_tasks: dict[int, asyncio.Task] = {}
_channel_sessions: dict[int, dict[str, Any]] = {}

START_CMD_RE = re.compile(r"<START_AGENT>\s*(\{.*?\})\s*</START_AGENT>", re.DOTALL)


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

    input_payload.append({
        "role": "system",
        "content": START_AGENT_SYSTEM_PROMPT,
    })

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


def _extract_start_command(text: str) -> tuple[Optional[dict], str]:
    match = START_CMD_RE.search(text or "")
    if not match:
        return None, text
    raw = match.group(1)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None, text
    cleaned = START_CMD_RE.sub("", text).strip()
    return payload, cleaned


def _append_operator_message(path: Path, session_id: str, author: str, content: str) -> None:
    payload = {
        "type": "operator_message",
        "session_id": session_id,
        "author": author,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "discord",
    }
    try:
        line = json.dumps(payload, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        return


def _append_operator_command(path: Path, session_id: str, author: str, command: str) -> None:
    payload = {
        "type": "operator_command",
        "session_id": session_id,
        "author": author,
        "command": command,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "discord",
    }
    try:
        line = json.dumps(payload, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        return


def _is_stop_command(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return normalized in {
        "stop",
        "!stop",
        "/stop",
        "cancel",
        "abort",
        "halt",
        "quit",
        "end",
        "terminate",
    }


async def _tail_step_log(path: Path, channel: discord.abc.Messageable, session_id: str, channel_id: int) -> None:
    last_pos = 0
    try:
        while True:
            if not path.exists():
                await asyncio.sleep(0.5)
                continue
            with path.open("r", encoding="utf-8") as f:
                f.seek(last_pos)
                lines = f.readlines()
                last_pos = f.tell()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if session_id and payload.get("session_id") != session_id:
                    continue
                evt = payload.get("type")
                if evt == "start":
                    msg = f"Started the agent with prompt: **{payload.get('task','').strip()}**"
                    await channel.send(msg)
                elif evt == "screenshot":
                    img_path = payload.get("path")
                    step = payload.get("step")
                    if img_path and Path(img_path).exists():
                        try:
                            await channel.send(
                                content=f"Step {step} screenshot:",
                                file=discord.File(str(img_path)),
                            )
                        except Exception:
                            pass
                    else:
                        await channel.send(f"Step {step} screenshot captured.")
                elif evt == "plan":
                    step = payload.get("step")
                    thought = payload.get("thought", "").strip()
                    action = payload.get("action", "").strip()
                    action_type = payload.get("action_type", "").strip()
                    parts = [
                        f"**Plan (Step {step})**",
                        f"**Action:** `{action_type}` - {action}" if action_type else f"**Action:** {action}" if action else None,
                        f"**Thought:** {thought}" if thought else None,
                    ]
                    msg = "\n".join([p for p in parts if p])
                    if msg:
                        await channel.send(msg)
                elif evt == "step":
                    thought = payload.get("thought", "").strip()
                    action = payload.get("action", "").strip()
                    action_type = payload.get("action_type", "").strip()
                    success = payload.get("success", False)
                    result = payload.get("result", "").strip()
                    status = "✅" if success else "❌"
                    parts = [
                        f"**Step {payload.get('step')}**",
                        f"**Action:** `{action_type}` - {action}" if action_type else f"**Action:** {action}",
                        f"**Thought:** {thought}" if thought else None,
                        f"**Result:** {status} {result}" if result else f"**Result:** {status}",
                    ]
                    msg = "\n".join([p for p in parts if p])
                    await channel.send(msg)
                elif evt == "complete":
                    msg = (
                        f"**Agent finished**\n"
                        f"Status: `{payload.get('status')}`\n"
                        f"Steps: `{payload.get('steps')}`\n"
                        f"Message: {payload.get('final_message','')}"
                    )
                    await channel.send(msg)
                    sess = _channel_sessions.get(channel_id)
                    if sess and sess.get("session_id") == session_id:
                        sess["active"] = False
                    return
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        return


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

    if DISCORD_TEST_CHANNEL_ID:
        try:
            allowed_channel_id = int(DISCORD_TEST_CHANNEL_ID)
        except ValueError:
            print("[discord] DISCORD_TEST_CHANNEL_ID must be an integer")
            return

        if message.channel.id != allowed_channel_id:
            return

    session = _channel_sessions.get(message.channel.id)
    if session and session.get("active") and session.get("message_log"):
        if _is_stop_command(message.content):
            _append_operator_command(
                Path(session["message_log"]),
                session.get("session_id", ""),
                message.author.display_name,
                "stop",
            )
            await message.channel.send("Stopping agent...")
            return
        _append_operator_message(
            Path(session["message_log"]),
            session.get("session_id", ""),
            message.author.display_name,
            message.content,
        )
        await message.channel.send("Added your note to the running agent.")
        return

    if not DISCORD_USE_MESSAGE_CONTENT and bot.user not in message.mentions:
        return

    history = _channel_histories.setdefault(message.channel.id, deque(maxlen=HISTORY_LIMIT))
    reply_text = await _generate_reply(message.content, list(history))
    start_payload, cleaned_reply = _extract_start_command(reply_text)
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": reply_text})

    if start_payload and isinstance(start_payload, dict):
        prompt = (start_payload.get("prompt") or "").strip()
        if not prompt:
            await message.channel.send("Agent start command received but prompt was empty.")
            return

        # Prepare step log path
        runs_dir = Path(__file__).resolve().parents[1] / "agent_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        session_id = f"{message.channel.id}-{int(message.created_at.timestamp())}"
        step_log = runs_dir / f"agent_steps_{session_id}.jsonl"
        message_log = runs_dir / f"agent_messages_{session_id}.jsonl"

        # Start GUI with auto-start prompt
        cmd = [
            "python",
            str(Path(__file__).resolve().parents[1] / "start_gui.py"),
            "--prompt",
            prompt,
            "--auto-start",
            "--step-log",
            str(step_log),
            "--message-log",
            str(message_log),
            "--session-id",
            session_id,
        ]
        subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]))

        # Start tailer for this channel
        existing = _channel_tail_tasks.get(message.channel.id)
        if existing and not existing.done():
            existing.cancel()
        _channel_sessions[message.channel.id] = {
            "session_id": session_id,
            "step_log": str(step_log),
            "message_log": str(message_log),
            "active": True,
        }
        _channel_tail_tasks[message.channel.id] = asyncio.create_task(
            _tail_step_log(step_log, message.channel, session_id, message.channel.id)
        )

        if cleaned_reply:
            for chunk in _chunk_text(cleaned_reply):
                await message.channel.send(chunk)
        return

    for chunk in _chunk_text(cleaned_reply if cleaned_reply else reply_text):
        await message.channel.send(chunk)


if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise SystemExit("Missing DISCORD_BOT_TOKEN in environment")

    bot.run(DISCORD_BOT_TOKEN)
