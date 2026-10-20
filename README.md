# Task Organizer

Turn a messy, natural-language description of your tasks into a prioritized to-do list.
Built with **LangGraph** (workflow) and **Grok (xAI)** or **Groq** (language understanding).

> "I need to finish my assignment tomorrow, buy groceries today, and prepare for my meeting next week."

```
📋 YOUR ORGANIZED TO-DO LIST
================================

🔴 HIGH PRIORITY
  [ ] Finish assignment (due: tomorrow)
      ↳ Deadline is tomorrow.

🟡 MEDIUM PRIORITY
  [ ] Buy groceries (due: today)
      ↳ Needed today but low stakes.

🟢 LOW PRIORITY
  [ ] Prepare for meeting (due: next week)
      ↳ Plenty of time remaining.
```

## Workflow

```
START → extract_tasks ─┬→ prioritize_tasks ⟲ (retry if any task is missing, max 3 tries)
                       │          │
                       │          ▼
                       └────→ format_results → END
```

| Node | Uses the LLM? | What it does |
|------|--------------|--------------|
| `extract_tasks` | Yes | Splits the message into individual tasks with optional deadlines |
| `prioritize_tasks` | Yes | Assigns High / Medium / Low with a one-line reason. Re-asks only for tasks it missed |
| `format_results` | No | Sorts by priority and renders the list. Anything still unprioritized defaults to Medium |

If no tasks are found, the graph skips straight to formatting and reports that.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # Windows: copy .env.example .env
# edit .env and paste ONE key. The provider is detected from the key prefix:
#   gsk_...  -> Groq     (free tier available)  https://console.groq.com/keys
#   xai-...  -> xAI Grok (paid)                 https://console.x.ai
# NOTE: Grok (xAI) and Groq are different companies, so their keys aren't interchangeable.
```

## Cute web UI 🎀

```bash
python -m task_organizer --serve
```

Opens http://127.0.0.1:8000 in your browser. Brain-dump your tasks, hit **organize me**, then tap
tasks to check them off (finish them all for confetti). It has example chips, dark mode, and a copy button.
Options: `--port 9000`, `--no-browser`. It runs only on your machine, and your API key never reaches the browser.

## CLI usage

```bash
# Pass text directly
python -m task_organizer "finish my assignment tomorrow, buy groceries today, prep for meeting next week"

# Interactive prompt
python -m task_organizer

# From a file or a pipe
python -m task_organizer --file my_tasks.txt
echo "call the bank today, renew passport someday" | python -m task_organizer

# Other formats, and save to a file
python -m task_organizer "..." --format markdown -o todo.md
python -m task_organizer "..." --format json

# Pick a different model
python -m task_organizer "..." --model openai/gpt-oss-20b

# Print the workflow as a Mermaid diagram
python -m task_organizer --graph
```

### Use it from Python

```python
from task_organizer import organize_tasks

state = organize_tasks("pay rent today, plan holiday next month")
print(state["formatted_output"])
print(state["final_tasks"])   # list of dicts: id, title, deadline, priority, reason
```

## Tests

The tests use a fake LLM, so they need no API key and make no network calls.

```bash
pytest
```

## Project layout

```
task_organizer/
  schemas.py     Pydantic models for the LLM's structured output
  state.py       LangGraph state definition
  llm.py         LLM client (Groq / xAI), retries, key detection
  nodes.py       The three nodes and the conditional routers
  graph.py       Graph assembly + organize_tasks()
  formatters.py  text / markdown / json renderers
  cli.py         Command-line interface
  web.py         Local web server for the UI (standard library only)
  static/        index.html: the cute UI
tests/           pytest suite with a fake LLM
```

## Notes

- Relative deadlines ("tomorrow") are resolved against today's date, which is injected into the prompts.
- Tasks are tracked by numeric id, not title, so the prioritizer can't rename or duplicate them.
- Exit codes: `0` success, `1` runtime/API error, `2` missing API key.

## Troubleshooting

- **"Incorrect API key"**: Grok (xAI, `xai-...`, paid) and Groq (`gsk_...`, free tier) are different
  services. The app auto-detects which one your key belongs to; if it still says rejected, create a
  fresh key, paste it into `.env` with no quotes or spaces, and restart. The error shows the key's
  first 4 characters and length so you can check what was loaded.
- **`503` / `504` / `429`**: The provider is busy or rate-limiting you. The app retries up to
  4 times per model (waiting 2s, 4s, 8s). Add backups with `LLM_FALLBACK_MODELS=model-a,model-b`.
- **`404` / model not found**: Model names change. Set `LLM_MODEL=<current name>` in `.env`
  (or pass `--model`). The default Groq model is `openai/gpt-oss-20b`.
  Groq models: https://console.groq.com/docs/models, xAI: https://docs.x.ai
