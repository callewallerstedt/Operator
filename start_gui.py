#!/usr/bin/env python3
"""
AI Computer Control Agent - GUI Launcher
Launch this to start the graphical interface.
"""

import sys
sys.path.insert(0, '.')

from agent.gui_app import launch_gui

if __name__ == "__main__":
    launch_gui()
