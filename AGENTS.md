# Repository Guidelines

## Project Structure & Module Organization
- `src/webapp`: FastAPI entry (`main.py`) plus templates/static for the UI.
- `src/pokedata_client`, `pricing`, `mapping`, `importer`, `exporter`, `risk`, `storage`, `utils`: API client, pricing logic, SKU mapping, Excel ingest/export, risk checks, local persistence, and helpers.
- `config/config.yaml`: FX and threshold config; `.env` for secrets.
- `data/input`, `data/output`, `data/mapping`, `logs`: runtime artifacts; keep out of git.
- `tests/`: unit tests; `test_api.py` is an additional API check at repo root.
- CI lives in `.github/workflows/ci.yml`; Docker build in `Dockerfile` and `docker-compose.yml`.

## Build, Test, and Development Commands
- Install dev deps: `python -m pip install --upgrade pip && pip install -e ".[dev]"` (or `pip install -r requirements.txt` for runtime-only).
- Local run: `uvicorn src.webapp.main:app --reload` (opens http://127.0.0.1:8000).
- Tests with coverage: `pytest tests/ -v --cov=src --cov-report=term-missing`; ensure `data/input data/output data/mapping logs` exist and set `POKEDATA_API_KEY` (use `test_key` for CI parity).
- Lint/format: `black .`, `isort .`, `ruff check .`.
- Type check: `mypy src/ --ignore-missing-imports`.
- Pre-commit (preferred before push): `pre-commit run --all-files`.
- Docker smoke: `docker build -t sitegiant-pricing:test . && docker run --rm sitegiant-pricing:test python -c "from src.main import main; print('Import OK')"`.

## Coding Style & Naming Conventions
- Python 3.10+; Black formatting (100 cols), isort Black profile, Ruff lint set (`E,W,F,I,B,C4,UP,SIM`), mypy optional typing (untyped defs allowed but checked).
- Use snake_case for functions/vars, PascalCase for classes, and lowercase modules/packages.
- Tests: files `test_*.py`, functions `test_*`; prefer descriptive fixtures and parametrize over loops.

## Testing Guidelines
- Primary framework: pytest with `--tb=short`, strict markers, coverage source `src`.
- Keep tests isolated from real services; mock external HTTP calls and FX lookups.
- Include regression fixtures under `tests/` and avoid writing to tracked paths (use `data/output`).
- Aim to keep coverage aligned with CI target (`--cov=src`); add focused unit tests for pricing, risk thresholds, and mapping edge cases.

## Commit & Pull Request Guidelines
- Use clear, present-tense messages; follow Conventional Commit style where possible (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`). One logical change per commit.
- Before opening a PR: run formatters, lint, mypy, and pytest; note any failures or intentional skips.
- PR description should include: purpose, key changes, manual test notes (commands + results), and any config/env impacts. Link issues or tickets; attach UI screenshots if the web UI changes.
- Do not commit secrets or sample data with real Pokedata IDs. Update `.env.example` and `config/config.yaml` if defaults change.

## Security & Configuration Tips
- Keep `.env` out of version control; use `.env.example` for placeholders.
- `.gitignore` must ignore all `.env` and `.env.*` files so API keys never enter the repo.
- When running locally, verify `fx` mode and thresholds in `config/config.yaml` before processing production exports.
- Prefer HTTP client timeouts and retries in new integrations; avoid adding wide `except` blocks around pricing logic without logging.
