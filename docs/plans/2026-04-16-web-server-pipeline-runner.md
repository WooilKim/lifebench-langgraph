# Web Server Pipeline Runner Implementation Plan

> For Hermes: Use subagent-driven-development or Claude Code to implement this plan task-by-task.

Goal: Add a local web server so users can start the LifeBench generation pipeline directly from the browser and inspect fresh results without manually running the CLI first.

Architecture: Keep the existing pipeline graph untouched. Add a thin Python web server layer that serves the current HTML viewer, exposes JSON APIs to launch and monitor pipeline runs, and derives viewer data from output folders on demand. Prefer a small server with explicit run-state management over embedding logic into the static page.

Tech Stack: Python 3, FastAPI (or an equally lightweight Python web framework if Claude finds a compelling reason), uvicorn, existing LangGraph pipeline code, minimal frontend JavaScript updates.

---

### Task 1: Inspect and document integration points

Objective: Identify the minimal set of files and data transformations needed for server-backed execution.

Files:
- Review: `run_full.py`
- Review: `pipeline/full_graph.py`
- Review: `explorer.html`
- Review: `index.html`
- Create or modify: `README.md`

Step 1: Confirm how pipeline state is initialized and saved today.
Step 2: Confirm how `explorer.html` currently loads `pipeline_data.json`.
Step 3: Decide which output directory the server should use by default.
Step 4: Document the chosen integration points in code comments and README updates.

Verification:
- The plan for server endpoints and data loading is explicit in code or README.

### Task 2: Extract reusable pipeline execution logic from the CLI

Objective: Make pipeline execution callable from both CLI and web server without duplicating logic.

Files:
- Modify: `run_full.py`
- Create: `server/pipeline_runner.py`
- Create: `server/__init__.py`
- Test: `tests/test_pipeline_runner.py`

Step 1: Write a failing test for pure helper logic such as result-summary building and output serialization path behavior.
Step 2: Run the targeted test and verify it fails.
Step 3: Extract reusable functions from `run_full.py`, such as:
- build initial state
- execute graph
- save per-persona outputs
- return a structured summary payload for API consumers
Step 4: Keep `run_full.py` as a thin CLI wrapper around the reusable module.
Step 5: Run targeted tests, then the full test suite.

Verification:
- `python3 run_full.py --count 1 --test` still works.
- Reusable execution function returns structured run metadata and output paths.

### Task 3: Build server-side data loader for the viewer

Objective: Replace the dependency on a checked-in static `pipeline_data.json` with server-generated JSON based on the current output directory.

Files:
- Create: `server/data_loader.py`
- Test: `tests/test_data_loader.py`

Step 1: Write failing tests for converting saved persona/output JSON files into the shape expected by `explorer.html`:
- `persona`
- `stats`
- `samples`
Step 2: Run the targeted tests and verify failure.
Step 3: Implement loader functions that read `metadata.json`, `persona.json`, `behavior_events*.json`, and build the viewer payload.
Step 4: Include gmail counts and transcript counts if present.
Step 5: Run tests.

Verification:
- Loader returns a list matching the current `pipeline_data.json` schema closely enough that the page can render without major rewrite.

### Task 4: Implement the web server and API endpoints

Objective: Expose browser-friendly endpoints for serving pages, reading current data, and triggering runs.

Files:
- Create: `server/app.py`
- Create: `server/run_manager.py`
- Test: `tests/test_server_api.py`
- Optionally create: `server/static/` only if needed; otherwise serve existing root HTML files directly.

Required endpoints:
- `GET /api/health` → basic readiness info
- `GET /api/data` → viewer payload generated from current output dir
- `POST /api/run` → start a pipeline run with JSON body like `{count, provider, output, test_mode}`
- `GET /api/run/{run_id}` → status, logs, summary, errors
- `GET /api/runs` → optional recent run list if easy

Implementation notes:
- Use background task or thread so the HTTP request returns quickly.
- Store in-memory run status plus basic logs.
- Do not modify the base LangGraph graph behavior.
- Default output directory can be something like `output_web/` or another server-specific location to avoid clobbering checked-in artifacts unless Claude determines reuse of `output_test/` is better and documents why.

Verification:
- Server starts locally.
- `curl` to health/data endpoints succeeds.
- A POST to `/api/run` returns a run id.
- Polling status eventually shows success or a useful error.

### Task 5: Update the frontend to launch runs from the browser

Objective: Add a simple control panel to the existing viewer so users can start pipeline generation and refresh results in place.

Files:
- Modify: `explorer.html`
- Optionally modify: `index.html`

Step 1: Add a small run form with fields:
- persona count
- provider (`claude|gpt|glm`)
- test mode checkbox
- output directory (optional advanced field)
Step 2: Replace static `fetch('./pipeline_data.json')` with API-first loading from `/api/data`, while preserving a graceful fallback if no data exists.
Step 3: Add run-status UI:
- idle
- running
- success
- failed
Step 4: After success, reload `/api/data` and rerender the page.
Step 5: Keep the page usable even before any run has been executed.

Verification:
- Opening the page on the local server shows the run controls.
- Starting a run updates status in the browser.
- On completion, newly generated personas appear without manual file refresh steps.

### Task 6: Document local usage

Objective: Make it easy to run the new server from scratch.

Files:
- Modify: `README.md`
- Modify: `requirements.txt` if new dependencies are added

Step 1: Document how to install dependencies.
Step 2: Document how to start the server, for example `uvicorn server.app:app --reload` or an equivalent command.
Step 3: Document required environment variables for LLM providers.
Step 4: Document the difference between CLI mode and web-server mode.

Verification:
- README contains a clear “Run the web server” section.

### Task 7: Final verification

Objective: Ensure the feature works end-to-end and does not regress the CLI.

Files:
- Verify repo-wide changes only; no new target file required

Step 1: Run focused tests.
Step 2: Run full test suite.
Step 3: Start the server locally.
Step 4: Hit API endpoints manually.
Step 5: Run one small end-to-end generation in test mode.
Step 6: Confirm the UI can display the generated results.

Verification:
- CLI still works.
- Web server works.
- Existing static viewer is adapted rather than abandoned.
- README instructions are accurate.
