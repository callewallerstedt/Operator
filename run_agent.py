#!/usr/bin/env python3
"""
AI Computer Control Agent - Main Entry Point

An intelligent AI agent that can control your computer to complete complex tasks
through vision, OCR, and intelligent action planning.

Usage:
    python run_agent.py "Open Excel and create a new spreadsheet"
    python run_agent.py --interactive
    python run_agent.py --task "Sort the files in Downloads folder by date"
"""

import sys
import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

# Add the agent module to path
sys.path.insert(0, '.')

from agent.config import config
from agent.loop import AgentLoop, AgentStatus
from agent.utils import format_duration, get_screen_resolution


console = Console()


def print_banner():
    """Print the startup banner."""
    banner = """
+===========================================================+
|                                                           |
|      AI COMPUTER CONTROL AGENT                            |
|                                                           |
|      Control your computer with natural language commands |
|      Powered by GPT-4 Vision                              |
|                                                           |
+===========================================================+
"""
    console.print(banner, style="bold cyan")


def print_status():
    """Print current system status."""
    table = Table(title="System Status", show_header=False, border_style="dim")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    width, height = get_screen_resolution()
    table.add_row("Screen Resolution", f"{width} x {height}")
    table.add_row("OpenAI Model", config.openai_model)
    table.add_row("Max Steps", str(config.max_steps))
    
    api_key_status = "Configured" if config.openai_api_key else "Missing"
    table.add_row("API Key", api_key_status)
    
    console.print(table)
    console.print()


def run_task(task: str):
    """Run a single task."""
    # Simple language detection for CLI
    try:
        import locale
        lang = locale.getdefaultlocale()[0]
        if lang and lang.startswith("sv"):
            system_language = "Swedish"
        elif lang and lang.startswith("de"):
            system_language = "German"
        elif lang and lang.startswith("fr"):
            system_language = "French"
        elif lang and lang.startswith("es"):
            system_language = "Spanish"
        else:
            system_language = "English"
    except Exception:
        system_language = "English"

    agent = AgentLoop()

    console.print()
    console.print(Panel(
        f"[bold]Task:[/bold] {task}\n[dim]System Language: {system_language}[/dim]",
        title="STARTING TASK",
        border_style="cyan"
    ))
    console.print()

    start_time = time.time()

    try:
        result = agent.run(task, system_language=system_language)
        
        # Print results
        console.print()
        
        if result.status == AgentStatus.COMPLETED:
            console.print(Panel(
                f"[green bold]TASK COMPLETED SUCCESSFULLY![/green bold]\n\n"
                f"[dim]Steps taken:[/dim] {result.steps_taken}\n"
                f"[dim]Duration:[/dim] {format_duration(result.duration)}\n\n"
                f"{result.final_message}",
                title="Result",
                border_style="green"
            ))
        elif result.status == AgentStatus.FAILED:
            console.print(Panel(
                f"[red bold]TASK FAILED[/red bold]\n\n"
                f"[dim]Steps taken:[/dim] {result.steps_taken}\n"
                f"[dim]Duration:[/dim] {format_duration(result.duration)}\n\n"
                f"{result.final_message}",
                title="Result",
                border_style="red"
            ))
        else:
            console.print(Panel(
                f"[yellow bold]TASK STOPPED[/yellow bold]\n\n"
                f"[dim]Steps taken:[/dim] {result.steps_taken}\n"
                f"[dim]Duration:[/dim] {format_duration(result.duration)}\n\n"
                f"{result.final_message}",
                title="Result",
                border_style="yellow"
            ))
        
        return result
        
    except Exception as e:
        console.print(f"[red bold]Error:[/red bold] {e}")
        raise


def interactive_mode():
    """Run in interactive mode, accepting commands one by one."""
    print_banner()
    print_status()
    
    console.print("[bold green]Interactive mode started![/bold green]")
    console.print("[dim]Type your tasks in natural language. Type 'quit' or 'exit' to stop.[/dim]")
    console.print("[dim]Press Ctrl+C during execution to stop the current task.[/dim]")
    console.print("[dim]A screenshot viewer window will open showing what the agent sees.[/dim]")
    console.print()
    
    while True:
        try:
            task = Prompt.ask("[bold cyan]Enter task[/bold cyan]")
            
            if task.lower() in ('quit', 'exit', 'q'):
                console.print("[dim]Goodbye![/dim]")
                break
            
            if task.lower() in ('help', '?'):
                print_help()
                continue
            
            if task.lower() == 'status':
                print_status()
                continue
            
            if not task.strip():
                continue
            
            console.print()
            run_task(task)
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted - task stopped[/yellow]")
            continue
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            continue


def print_help():
    """Print help information."""
    help_text = """
[bold cyan]AI Computer Control Agent - Help[/bold cyan]

[bold]Example Tasks:[/bold]
  • "Open Notepad and type 'Hello World'"
  • "Open Excel and create a new spreadsheet"
  • "Sort the files in Downloads folder by date"
  • "Open Chrome and go to google.com"
  • "Take a screenshot and save it to Desktop"
  • "Open File Explorer and navigate to Documents"
  • "Open Settings and change the wallpaper"

[bold]Commands:[/bold]
  • help, ?     - Show this help
  • status      - Show system status
  • quit, exit  - Exit the agent

[bold]Tips:[/bold]
  • Be specific about what you want
  • The agent uses keyboard shortcuts when possible (faster)
  • Press Ctrl+C to stop a running task
  • Complex tasks may take multiple steps
"""
    console.print(help_text)


def main():
    parser = argparse.ArgumentParser(
        description="AI Computer Control Agent - Control your computer with natural language"
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Task to execute (e.g., 'Open Excel and create a new spreadsheet')"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "-t", "--task",
        dest="task_flag",
        help="Task to execute (alternative to positional argument)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=f"Maximum steps before stopping (default: {config.max_steps})"
    )
    parser.add_argument(
        "-g", "--gui",
        action="store_true",
        help="Launch the graphical interface"
    )
    
    args = parser.parse_args()
    
    # Override config if needed
    if args.max_steps:
        config.max_steps = args.max_steps
    
    # Validate API key early
    try:
        config.validate()
    except ValueError as e:
        console.print(Panel(
            str(e),
            title="CONFIGURATION ERROR",
            border_style="red"
        ))
        console.print("\n[yellow]Create a .env file in the project root with:[/yellow]")
        console.print("OPENAI_API_KEY=your-api-key-here")
        sys.exit(1)
    
    # Determine what to do
    task = args.task or args.task_flag
    
    if args.gui:
        # Launch GUI mode
        from agent.gui_app import launch_gui
        launch_gui()
    elif args.interactive or not task:
        interactive_mode()
    else:
        print_banner()
        result = run_task(task)
        sys.exit(0 if result.status == AgentStatus.COMPLETED else 1)


if __name__ == "__main__":
    main()
