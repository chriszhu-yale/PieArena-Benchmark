# lm_human — Human-in-the-loop negotiation web app

A FastAPI backend and a static HTML frontend for running human practice sessions against the negotiation agents used in PieArena.

## Backend

```bash
export OPENROUTER_API_KEY=...  # or OPENAI_API_KEY
export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # optional; this is the default
export SCENARIOS_DIR=/absolute/path/to/scenarios
export DB_PATH=/absolute/path/to/negotiations.db

uvicorn main:app --reload --port 8000
```

Sessions are stored in the SQLite file at `DB_PATH`. The `/download_db?secret=...` endpoint is disabled unless `DOWNLOAD_KEY` is set. The app uses the shared client in `piearena/openai_wrapper.py` (run it from this folder inside the repository).

## Frontend

Serve `lm_human/` with any static server:

```bash
cd lm_human
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser. The UI uses `http://localhost:8000` as its backend by default; to point at a different backend:

```
http://localhost:8080/?api=http://your-backend:8000
```

This override is stored in browser localStorage under the key `negotiation_api_base`.

## Notes

- The API root (`/`) returns JSON; the UI is served separately as static HTML.
- If you see "Scenario file not found", verify `SCENARIOS_DIR` points to the directory of YAMLs.
