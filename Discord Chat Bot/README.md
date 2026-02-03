# Discord Chat Bot (OpenAI)

A minimal Discord bot that replies to incoming messages using the OpenAI API.

## What this does

- Listens for new messages
- Sends the message text to OpenAI
- Replies in the same channel
- Keeps a short memory of the last 6 messages per channel

## Setup

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Create `.env` in the project root

Copy `.env.example` to `c:\Cursor\Operator\.env` and fill in your values.

```bash
cp .env.example .env
```

3) Enable message content intent

In the Discord Developer Portal, enable **Message Content Intent** for your bot.
If you prefer not to enable it, set `DISCORD_USE_MESSAGE_CONTENT=false` and the bot
will only respond when mentioned.

4) Run the bot

```bash
python app.py
```

## Environment variables

- `OPENAI_API_KEY` (required)
- `CHAT_MODEL` (optional, default `gpt-4o-mini`)
- `DISCORD_BOT_TOKEN` (required)
- `DISCORD_TEST_CHANNEL_ID` (optional, send a startup test message)
- `DISCORD_USE_MESSAGE_CONTENT` (optional, default `true`)
- `CHAT_SYSTEM_PROMPT` (optional, system prompt for the bot)

