#!/usr/bin/env python3
"""
AI Computer Control Agent - GUI Launcher
Launch this to start the graphical interface.
"""

import sys
import os
import argparse
sys.path.insert(0, '.')

from agent.gui_app import launch_gui

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the GUI (optional auto-start).")
    parser.add_argument("--prompt", default="", help="Initial task prompt for the GUI")
    parser.add_argument("--auto-start", action="store_true", help="Auto-start the task on launch")
    parser.add_argument("--step-log", default="", help="Path to JSONL step log output")
    parser.add_argument("--message-log", default="", help="Path to JSONL operator message input")
    parser.add_argument("--session-id", default="", help="Session id for correlating logs")
    args = parser.parse_args()

    if args.prompt:
        os.environ["AGENT_PROMPT"] = args.prompt
    if args.auto_start:
        os.environ["AGENT_AUTO_START"] = "1"
    if args.step_log:
        os.environ["AGENT_STEP_LOG"] = args.step_log
    if args.message_log:
        os.environ["AGENT_MESSAGE_LOG"] = args.message_log
    if args.session_id:
        os.environ["AGENT_SESSION_ID"] = args.session_id

    launch_gui()
