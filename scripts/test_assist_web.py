#!/usr/bin/env python3
"""
Test helper for the Operator Assist web flow.

Flow:
1) Upload an image to a Discord channel (or use a provided image URL).
2) Send an Assist link to the channel.
3) (Optional) Wait for the !click message posted by the assist web.

Env:
- DISCORD_BOT_TOKEN
- DISCORD_TEST_CHANNEL_ID (or use --channel)
- ASSIST_WEB_URL (deployed assist_web URL)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
import mimetypes
import tempfile
import urllib.request
import urllib.error
from urllib.parse import quote


def _load_env() -> None:
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _request(
    method: str,
    url: str,
    token: str | None,
    body: bytes | None = None,
    content_type: str | None = None,
):
    req = urllib.request.Request(url, method=method)
    req.add_header("User-Agent", "OperatorAssistTest/1.0")
    if token:
        req.add_header("Authorization", f"Bot {token}")
    if content_type:
        req.add_header("Content-Type", content_type)
    if body is not None:
        req.data = body
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except Exception:
            raw = b""
        return exc.code, raw


def _send_json(token: str, channel_id: str, payload: dict) -> dict:
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    body = json.dumps(payload).encode("utf-8")
    status, raw = _request("POST", url, token, body, "application/json")
    if status >= 300:
        raise RuntimeError(f"Discord API error ({status}): {raw[:400]!r}")
    return json.loads(raw.decode("utf-8"))


def _build_multipart(payload_json: str, file_path: str) -> tuple[bytes, str]:
    boundary = f"----OperatorAssist{uuid.uuid4().hex}"
    parts: list[bytes] = []

    def add_part(headers: list[str], data: bytes) -> None:
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append("\r\n".join(headers).encode("utf-8"))
        parts.append(b"\r\n\r\n")
        parts.append(data)
        parts.append(b"\r\n")

    add_part([
        'Content-Disposition: form-data; name="payload_json"',
        "Content-Type: application/json",
    ], payload_json.encode("utf-8"))

    filename = os.path.basename(file_path)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    with open(file_path, "rb") as f:
        data = f.read()
    add_part([
        f'Content-Disposition: form-data; name="files[0]"; filename="{filename}"',
        f"Content-Type: {content_type}",
    ], data)

    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _send_with_file(token: str, channel_id: str, content: str, file_path: str) -> dict:
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    payload_json = json.dumps({"content": content})
    body, content_type = _build_multipart(payload_json, file_path)
    status, raw = _request("POST", url, token, body, content_type)
    if status >= 300:
        raise RuntimeError(f"Discord API error ({status}): {raw[:400]!r}")
    return json.loads(raw.decode("utf-8"))


def _send_webhook_json(webhook_url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    status, raw = _request("POST", webhook_url, None, body, "application/json")
    if status >= 300:
        raise RuntimeError(f"Discord webhook error ({status}): {raw[:400]!r}")
    return json.loads(raw.decode("utf-8")) if raw else {}


def _send_webhook_with_file(webhook_url: str, content: str, file_path: str) -> dict:
    payload_json = json.dumps({"content": content})
    body, content_type = _build_multipart(payload_json, file_path)
    status, raw = _request("POST", webhook_url, None, body, content_type)
    if status >= 300:
        raise RuntimeError(f"Discord webhook error ({status}): {raw[:400]!r}")
    return json.loads(raw.decode("utf-8")) if raw else {}


def _fetch_messages(token: str, channel_id: str, after: str | None = None, limit: int = 20) -> list[dict]:
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages?limit={limit}"
    if after:
        url += f"&after={after}"
    status, raw = _request("GET", url, token)
    if status >= 300:
        raise RuntimeError(f"Discord API error ({status}): {raw[:200]!r}")
    return json.loads(raw.decode("utf-8"))


def _parse_click(content: str, session_id: str | None) -> tuple[int, int] | None:
    if not content:
        return None
    parts = content.strip().split()
    if not parts or parts[0].lower() not in {"!click", "/click", "click"}:
        return None
    if len(parts) < 3:
        return None
    try:
        x = int(parts[1])
        y = int(parts[2])
    except ValueError:
        return None
    if session_id and len(parts) >= 4 and parts[3] != session_id:
        return None
    return x, y


def main() -> int:
    _load_env()

    parser = argparse.ArgumentParser(description="Send an Assist link to Discord for testing.")
    parser.add_argument("--channel", default=os.getenv("DISCORD_TEST_CHANNEL_ID", ""))
    parser.add_argument("--token", default=os.getenv("DISCORD_BOT_TOKEN", ""))
    parser.add_argument("--webhook", default=os.getenv("DISCORD_WEBHOOK_URL", ""))
    parser.add_argument("--assist-url", default=os.getenv("ASSIST_WEB_URL", ""))
    parser.add_argument(
        "--image",
        dest="image_path",
        default=os.path.join("images", "tesla.png"),
        help="Local image path to upload",
    )
    parser.add_argument("--image-url", default="", help="Public image URL (skip upload)")
    parser.add_argument("--screenshot", action="store_true", help="Capture a monitor screenshot instead of using --image")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index for screenshot (1=primary)")
    parser.add_argument("--left", type=int, default=0)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--step", default="1")
    parser.add_argument("--session", default="")
    parser.add_argument("--wait-click", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    if not args.assist_url:
        print("Missing ASSIST_WEB_URL", file=sys.stderr)
        return 2
    if not args.webhook and (not args.channel or not args.token):
        print("Missing DISCORD_TEST_CHANNEL_ID or DISCORD_BOT_TOKEN (or provide DISCORD_WEBHOOK_URL)", file=sys.stderr)
        return 2

    session_id = args.session or f"test-{int(time.time())}"

    attachment_url = args.image_url
    width = 0
    height = 0

    if args.screenshot:
        args.image_path = ""

    if not attachment_url and not args.image_path:
        try:
            import mss
            from PIL import Image
        except Exception as exc:
            print(f"Missing dependency for screenshot capture: {exc}", file=sys.stderr)
            return 2
        with mss.mss() as sct:
            monitor_index = max(1, int(args.monitor))
            if monitor_index >= len(sct.monitors):
                print(f"Monitor index {monitor_index} not available. Found {len(sct.monitors)-1} monitors.", file=sys.stderr)
                return 2
            mon = sct.monitors[monitor_index]
            screenshot = sct.grab(mon)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        tmp = tempfile.NamedTemporaryFile(prefix="assist_shot_", suffix=".png", delete=False)
        tmp.close()
        img.save(tmp.name, format="PNG")
        args.image_path = tmp.name
        args.left = int(mon.get("left", 0))
        args.top = int(mon.get("top", 0))

    if not attachment_url:
        if not args.image_path:
            print("Provide --image or --image-url", file=sys.stderr)
            return 2
        if args.webhook:
            resp = _send_webhook_with_file(
                args.webhook,
                f"Assist test image (session {session_id}).",
                args.image_path,
            )
        else:
            resp = _send_with_file(
                args.token,
                args.channel,
                f"Assist test image (session {session_id}).",
                args.image_path,
            )
        attachments = resp.get("attachments", [])
        if not attachments:
            print("No attachment URL returned by Discord.", file=sys.stderr)
            return 2
        attachment = attachments[0]
        attachment_url = attachment.get("url", "")
        width = int(attachment.get("width") or 0)
        height = int(attachment.get("height") or 0)

    assist_url = (
        f"{args.assist_url}"
        f"?img={quote(attachment_url, safe='')}"
        f"&w={width}&h={height}"
        f"&left={args.left}&top={args.top}"
        f"&session={session_id}&step={args.step}"
    )

    if args.webhook:
        _send_webhook_json(args.webhook, {
            "content": (
                "**Assist test**\n"
                f"Session: `{session_id}`\n"
                f"Open the assist page: {assist_url}"
            )
        })
    else:
        _send_json(args.token, args.channel, {
            "content": (
                "**Assist test**\n"
                f"Session: `{session_id}`\n"
                f"Open the assist page: {assist_url}"
            )
        })

    print("Assist link sent.")
    print(f"Session: {session_id}")
    print(f"Assist URL: {assist_url}")

    if not args.wait_click:
        return 0
    if not args.token or not args.channel:
        print("Cannot wait for click without DISCORD_BOT_TOKEN and DISCORD_TEST_CHANNEL_ID.", file=sys.stderr)
        return 2

    print("Waiting for !click message...")
    last_id: str | None = None
    start = time.time()
    while time.time() - start < args.timeout:
        messages = _fetch_messages(args.token, args.channel, after=last_id)
        if messages:
            # Discord returns newest first
            for msg in reversed(messages):
                last_id = msg.get("id", last_id)
                hit = _parse_click(msg.get("content", ""), session_id)
                if hit:
                    print(f"Click received: {hit[0]}, {hit[1]}")
                    return 0
        time.sleep(2)

    print("Timed out waiting for click.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
