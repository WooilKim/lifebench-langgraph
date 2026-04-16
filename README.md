# LifeBench LangGraph Pipeline

Korean-localized reimplementation of the LifeBench phone-behavior dataset,
rebuilt on top of LangGraph so every generation step is a swappable node.
See [CLAUDE.md](CLAUDE.md) for the full project charter.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set at least one LLM provider key:

```bash
export ANTHROPIC_API_KEY=...        # provider=claude (default)
export OPENAI_API_KEY=...            # provider=gpt
export ZHIPUAI_API_KEY=...           # provider=glm
```

## Run the CLI

```bash
python3 run_full.py --count 2 --provider claude
python3 run_full.py --count 1 --provider claude --test   # fast test mode
```

Per-persona JSON files and `metadata.json` are written under `--output`
(defaults to `output/`).

## Run the web server

The web server wraps the same pipeline logic so you can start runs from
the browser and inspect fresh results without re-running the CLI. It serves
only the viewer assets (`index.html`, `explorer.html`, `pipeline_data.json`) plus
JSON API endpoints — it does not expose the whole repository as static files.

```bash
python3 -m server.app --port 8765 --output output_web
# → http://127.0.0.1:8765/explorer.html
```

Flags:

- `--host` (default `127.0.0.1`)
- `--port` (default `8765`)
- `--output` — directory served by `/api/data` and used as the default
  target for `POST /api/run` (default `output_web`).

The same LLM provider env vars apply. The server relies only on the
Python standard library and LangGraph/LangChain — no extra web framework
dependencies are added.

### API endpoints

| Method | Path                  | Purpose                                               |
| ------ | --------------------- | ----------------------------------------------------- |
| GET    | `/api/health`         | Readiness info + current output dir                   |
| GET    | `/api/data`           | Viewer payload built from `--output` on demand         |
| POST   | `/api/run`            | Launch a pipeline run (JSON body, see below)          |
| GET    | `/api/run/{run_id}`   | Status, logs, summary, error                          |
| GET    | `/api/runs`           | Recent run list (most-recent-last)                    |

`POST /api/run` body:

```json
{ "count": 1, "provider": "claude", "test_mode": true, "output": "output_web" }
```

`count` (int, required-ish, defaults to 1), `provider` ∈
`claude|gpt|glm`, `test_mode` (bool), `output` (optional override of the
server-default output dir).

### Browser UI

`explorer.html` now loads from `/api/data` first and falls back to the
checked-in `pipeline_data.json` if no server is present. A small run
control panel at the top of the page lets you pick count/provider/
test-mode and start a run; status + logs update in place, the active
output directory is shown under the controls, and on success `/api/data`
is re-fetched so the new personas appear immediately. For safety,
web-triggered runs must write under the server's parent output directory
(for example `output_web/...`) rather than arbitrary filesystem paths.

## CLI vs web mode

Both paths call the same `server.pipeline_runner.run_pipeline` helper —
the CLI is just a thin argparse wrapper and the web server is just a
thin HTTP wrapper. Switching between them never changes the underlying
LangGraph graph or the per-persona output layout.

## Tests

```bash
python3 -m unittest discover -s tests
```

The test suite exercises the pipeline-runner helpers, the viewer data
loader, the run manager, and the HTTP API (started on an ephemeral
port with an injected fake runner so the tests are fast and don't hit
any LLM).
