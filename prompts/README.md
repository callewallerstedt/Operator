# Prompts

This folder contains the system prompts used by the different AI components.

Edit the `.txt` files to change behavior. Files may include placeholders like `{width}` or `{target_prompt}`; those are filled in at runtime.

## Structure
- `prompts/planner/system_prompt.txt`: Main agent planner system prompt.
- `prompts/planner/user_message.txt`: Planner user message template (state/context).
- `prompts/planner/verify_action_user.txt`: Planner verification prompt template.
- `prompts/coordinate_finder/*.txt`: Coordinate finder sub-agent system prompts.
- `prompts/coordinate_finder/planner_user.txt`: Coordinate finder planner user prompt template.
- `prompts/coordinate_finder/picker_user.txt`: Coordinate finder picker user prompt template.
- `prompts/discord_bot/system_prompt.txt`: Optional Discord chat bot system prompt (leave empty to disable).
- `prompts/discord_bot/start_agent_system.txt`: System prompt that enforces the `<START_AGENT>` command format.
