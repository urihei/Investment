# Investment Portfolio App

## What this repo needs to run

Required runtime files:

- `app.py`
- `portfolio_engine.py`
- `static/index.html`
- `data/portfolio.json`
- `data/fees.json`
- `requirements.txt`

Generated at runtime and **not required to commit**:

- `data/price_cache.json`
- `__pycache__/`
- `venv/`

Optional / not required for the Flask app:

- `main.py`
- `Clustering.py`
- files under `old/`
- ad-hoc CSV / XLSX files in `data/`

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

## Run

```bash
python app.py
```

Then open:

- `http://localhost:5000`

## Notes

- The app uses `yfinance` for market data where available.
- Some TASE assets use `CSV:` virtual tickers and are auto-fetched from saved TASE URLs in `data/portfolio.json`.
- The TASE market graph flow uses Playwright because some pages are JS-rendered / bot-protected.

