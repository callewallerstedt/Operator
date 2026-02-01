# AI Computer Control Agent

An intelligent AI agent that can fully control your Windows computer to complete complex tasks through vision, OCR, and intelligent action planning. Just describe what you want in natural language, and the agent will figure out how to do it.

## Features

- **Vision-Based Understanding**: Uses GPT-4 Vision to analyze your screen and understand the current state
- **OCR Text Detection**: Finds and clicks on any visible text using Tesseract OCR
- **Smart Action Planning**: Prioritizes keyboard shortcuts and efficient action chains
- **Command Chaining**: Executes sequences like "open start menu + type + enter" in one step
- **Multi-Language Support**: Detects system language (Swedish, German, etc.) for correct UI interaction
- **Multi-Monitor Support**: Select which monitor to capture
- **GUI Application**: Full graphical interface with live preview and settings
- **Real-time Monitoring**: See exactly what the AI sees at each step

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

You also need Tesseract OCR:
- **Windows**: Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Install to default location (C:\Program Files\Tesseract-OCR)

### 2. Configure API Key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

Or set environment variable:

```powershell
$env:OPENAI_API_KEY = "sk-your-openai-api-key-here"
```

### 3. Run the Agent

**GUI Mode (Recommended):**
```bash
python start_gui.py
```
or
```bash
python run_agent.py --gui
```

**Command Line:**
```bash
python run_agent.py "Open Calculator"
```

## GUI Features

The graphical interface provides:

- **Task Input**: Enter tasks in natural language
- **Monitor Selection**: Choose which screen to capture
- **Settings Panel**:
  - Action delay (ms between actions)
  - Click delay (ms after clicks)
  - Max steps before stopping
  - Command chaining toggle
  - Action verification toggle
  - Live preview toggle
- **Live Preview**: See the screen in real-time
- **Activity Log**: Colored output showing agent's progress

## Example Tasks

```
Open Notepad and type Hello World
Open Calculator
Open File Explorer and navigate to Downloads
Search for weather in the Start menu
Open Chrome and go to google.com
Create a new folder on the Desktop called Test
```

## Architecture

```
+--------------------------------------------------+
|                  GUI Application                  |
|  - Task input                                    |
|  - Monitor selection                             |
|  - Settings panel                                |
|  - Live screenshot viewer                        |
|  - Activity log                                  |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|                 Agent Loop                        |
|  1. Capture screenshot from selected monitor     |
|  2. Run OCR for context                          |
|  3. Send to Planner LLM                          |
|  4. Execute action (or chain of actions)         |
|  5. Wait for UI response                         |
|  6. Verify result                                |
|  7. Repeat until done                            |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|                 Planner LLM                       |
|  - Analyzes screenshot with GPT-4 Vision         |
|  - Considers system language                     |
|  - Outputs efficient action chains               |
|  - Prioritizes keyboard shortcuts                |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|                  Executors                        |
|  - Keyboard (hotkeys, typing)                    |
|  - Mouse (click, scroll, drag)                   |
|  - OCR Click (find text and click)               |
|  - Chain (multiple actions with delays)          |
|  - Vision Click (marker-based fallback)          |
+--------------------------------------------------+
```

## Project Structure

```
agent/
  __init__.py        # Package init
  config.py          # Configuration management
  screenshot.py      # DPI-aware screen capture
  ocr.py             # Tesseract OCR integration
  actions.py         # Action schemas (including chains)
  executors.py       # Keyboard, mouse, OCR click
  planner.py         # GPT-4 Vision planner
  vision_click.py    # Visual marker click fallback
  loop.py            # Main agent orchestration
  gui_app.py         # GUI application
  utils.py           # Utility functions

start_gui.py         # GUI launcher
run_agent.py         # CLI entry point
requirements.txt     # Dependencies
.env                 # API key (create this)
```

## Configuration

Settings can be adjusted in the GUI or in `agent/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `post_action_delay` | 0.8s | Delay after actions |
| `click_delay` | 0.3s | Delay after clicks |
| `max_steps` | 50 | Maximum steps |
| `openai_model` | gpt-4o | Vision model |

## Multi-Language Support

The agent detects your system language and accounts for UI elements being in that language:

- Swedish: Arkiv (File), Redigera (Edit), Installningar (Settings)
- German: Datei (File), Bearbeiten (Edit), Einstellungen (Settings)
- And more...

## Safety

- The agent controls your real computer
- It moves your mouse and types on your keyboard
- Press the Stop button or close the app to halt execution
- Always save your work before running complex tasks

## Troubleshooting

**"Tesseract not found"**
- Install from the link above
- Should be at: C:\Program Files\Tesseract-OCR\tesseract.exe

**"API key not found"**
- Create `.env` file with OPENAI_API_KEY=your-key
- Or set environment variable

**Agent not seeing correct screen**
- Use the Monitor dropdown to select the right display

**Actions too fast/slow**
- Adjust delays in Settings panel

## License

MIT License
