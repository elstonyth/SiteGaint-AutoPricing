# SiteGiant Pricing Automation

Sync Pokedata prices to SiteGiant Webstore with a guided workflow and built-in safety checks.

## What This App Does
- Fetches live prices from the Pokedata API (English & Japanese products).
- Converts USD→MYR with Google Finance or a manual rate.
- Flags risky price changes with soft/hard thresholds before you export.
- Preserves SiteGiant Excel format so you can re-import immediately.

## Requirements
- Python 3.10+ (Windows 10/11 recommended; works in a virtualenv).
- Pokedata API key (add it to `.env`; see Setup).

## Setup
1) Create a virtual environment  
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux
```

2) Install dependencies  
```bash
pip install -r requirements.txt
```

3) Add your API key (never commit it)  
```bash
copy .env.example .env   # or cp .env.example .env
```
Edit `.env` and set `POKEDATA_API_KEY=your_key_here`. `.gitignore` already excludes `.env` and `.env.*`.

4) Create the desktop shortcut (optional, Windows)  
```powershell
powershell -ExecutionPolicy Bypass -File create-shortcut.ps1
```

## Running the App
- One-click (Windows): double-click **SiteGiant Pricing** on your Desktop (or run `launch.bat`).
- Manual: `uvicorn src.webapp.main:app --reload`

The UI opens at http://127.0.0.1:8000

## Typical Workflow
1. Go to **Settings → Mapping** and upload your SKU ↔ Pokedata mapping (one-time, update as needed).
2. Export products from SiteGiant Webstore and place the file in `data/input/`.
3. In **Price Update**, upload the Excel file, review prices and risk warnings, then select items to export.
4. Download the updated Excel from `data/output/` and import back into SiteGiant.

## Mapping File Format
Required columns: `sku`, `pokedata_id`, `pokedata_language` (`ENGLISH`|`JAPANESE`), `auto_update` (`Y`|`N`)  
Optional: `isku`, `name`, `pokedata_name`, `pokedata_url`, `notes`

Example:
| sku | isku | name | pokedata_id | pokedata_name | pokedata_language | auto_update |
|-----|------|------|-------------|---------------|-------------------|-------------|
| PKM-EN-001 | INT-001 | SV Booster Box | 66 | Scarlet & Violet BB | ENGLISH | Y |
| PKM-JP-001 | INT-002 | 151 Japanese Box | 89 | Pokemon 151 BB | JAPANESE | Y |

## Pricing Formula
```
Final MYR Price = (USD Price × FX Rate) ÷ 0.8
```
Example: 89.65 USD × 4.70 ÷ 0.8 = RM 526.70

## Configuration
- `.env` (secrets — never commit): `POKEDATA_API_KEY`, optional `LOG_LEVEL`, `LOG_FORMAT`.
- `config/config.yaml`: FX mode (`google` or `manual`), default FX rate, pricing divisor, risk thresholds, paths, caching, and Pokedata settings. Adjust before production runs if needed.

## Project Structure (high level)
- `src/webapp`: FastAPI app and UI templates/static.
- `src/pokedata_client`: Pokedata API client with retries and timeouts.
- `src/pricing`, `mapping`, `importer`, `exporter`, `risk`: pricing logic, mapping utilities, Excel ingest/export, threshold checks.
- `data/input`, `data/output`, `data/mapping`, `logs`: runtime files (kept out of git).

## Running Tests (optional)
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```
Set `POKEDATA_API_KEY=test_key` for parity with CI.

## Docker (optional)
```bash
docker build -t sitegiant-pricing .
docker run -p 8000:8000 sitegiant-pricing
```

## Safety & Privacy
- API keys live only in `.env`; `.gitignore` blocks `.env` and `.env.*`.
- Do not put real keys in tracked files; share defaults via `.env.example` if needed.
- Runtime artifacts stay in `data/` and `logs/`; keep them out of version control.

## Troubleshooting
- **No mapping file**: upload a mapping in Settings → Mapping first.
- **Demo mode warning**: set `POKEDATA_API_KEY` in `.env` and restart.
- **FX rate not updating**: check connectivity; the app falls back to `default_rate` in `config/config.yaml`.

## License
Private - Internal use only
