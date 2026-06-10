"""
Flask API for Investment Portfolio Monitor
==========================================
Endpoints:
  GET  /                              → serve index.html
  GET  /api/portfolio/timeline        → full portfolio value over time
  GET  /api/portfolio/summary         → current holdings snapshot
  GET  /api/portfolio/fees            → all fees & taxes
  GET  /api/portfolio/gains           → per-asset gain summary
  GET  /api/portfolio/yearly_yield    → yearly return metrics (TWR/XIRR/dividend yield)
  GET  /api/portfolio/yield_series    → rolling return points (weekly/monthly)
  GET  /api/asset/<name>/history      → single asset value history
  GET  /api/actions                   → list all actions
  POST /api/actions                   → add a new action
  PUT  /api/actions/<int:idx>         → update an action
  DELETE /api/actions/<int:idx>       → delete an action
  GET  /api/currencies                → supported display currencies

Query params accepted by most GET endpoints:
  currency=ILS|USD|EUR|GBP  (default: ILS)
  frequency=D|W|M            (default: W)
"""

import csv
import io
import json
import logging
import re
import threading
from datetime import date, datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from portfolio_engine import PortfolioEngine, PORTFOLIO_JSON, FEES_JSON, CACHE_JSON

# ── App setup ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

SUPPORTED_CURRENCIES = ["ILS", "USD", "EUR", "GBP"]

# Singleton engine — loaded once at startup
_engine: PortfolioEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> PortfolioEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                logger.info("Loading PortfolioEngine...")
                _engine = PortfolioEngine(PORTFOLIO_JSON, FEES_JSON, CACHE_JSON)
                logger.info("PortfolioEngine loaded. Pre-warming cache...")
                _prewarm(_engine)
    return _engine


def _prewarm(engine: PortfolioEngine):
    """
    Step 3 — Pre-fetch all ticker price/dividend/FX histories in background
    so the first real request is fast.
    """
    def run():
        try:
            # Collect all unique tickers
            tickers = set()
            for info in engine.initial_holdings.values():
                t = info.get("ticker")
                if t:
                    tickers.add((t, info.get("currency") or "ILS"))
            for action in engine.actions:
                t = action.get("ticker")
                if t:
                    tickers.add((t, action.get("currency") or "ILS"))

            start = engine.start_date
            end   = engine.end_date

            for ticker, currency in tickers:
                logger.info("Pre-warming price: %s", ticker)
                engine.fetch_price_history(ticker, start, end)
                engine.fetch_dividend_history(ticker, start, end)
                # FX: ticker native → ILS
                tc = engine.get_ticker_currency(ticker)
                if tc != "ILS":
                    engine.fetch_exchange_rate_history(tc, "ILS", start, end)

            # Common FX pairs
            for pair in [("USD", "ILS"), ("EUR", "ILS"), ("GBP", "ILS")]:
                engine.fetch_exchange_rate_history(pair[0], pair[1], start, end)

            logger.info("Cache pre-warm complete.")
        except Exception as e:
            logger.error("Pre-warm failed: %s", e)

    t = threading.Thread(target=run, daemon=True)
    t.start()


# ── Helpers ────────────────────────────────────────────────────────────────

def _currency():
    c = request.args.get("currency") or "ILS".upper()
    return c if c in SUPPORTED_CURRENCIES else "ILS"


def _frequency():
    f = request.args.get("frequency", "W").upper()
    return f if f in ("D", "W", "M") else "W"


def _reload_engine():
    """Force engine reload (called after actions are modified)."""
    global _engine
    with _engine_lock:
        _engine = None


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/currencies")
def get_currencies():
    return jsonify({"currencies": SUPPORTED_CURRENCIES, "default": "ILS"})


@app.route("/api/portfolio/timeline")
def portfolio_timeline():
    """Full portfolio value over time."""
    eng = get_engine()
    result = eng.get_portfolio_timeline(
        display_currency=_currency(),
        frequency=_frequency(),
    )
    return jsonify(result)


@app.route("/api/portfolio/summary")
def portfolio_summary():
    """Current holdings snapshot with latest prices."""
    eng = get_engine()
    result = eng.get_current_holdings_summary(display_currency=_currency())
    return jsonify(result)


@app.route("/api/portfolio/fees")
def portfolio_fees():
    """All fees and taxes breakdown."""
    eng = get_engine()
    result = eng.calculate_fees_and_taxes(display_currency=_currency())
    return jsonify(result)


@app.route("/api/portfolio/gains")
def portfolio_gains():
    """Per-asset gain summary for all holdings."""
    eng = get_engine()
    currency = _currency()
    gains = []

    all_infos = dict(eng.initial_holdings)
    for action in eng.actions:
        sym = action.get("symbol")
        if action["action_type"] == "buy" and sym and sym not in all_infos:
            all_infos[sym] = {
                "quantity": 0.0,
                "ticker": action.get("ticker"),
                "currency": action.get("currency") or "ILS",
                "asset_type": "stock",
            }

    for name, info in all_infos.items():
        try:
            g = eng.get_asset_gain_summary(name, info, currency)
            if g["current_quantity"] > 0 or g["cost_basis"] > 0:
                gains.append(g)
        except Exception as e:
            logger.error("Gain summary failed for %s: %s", name, e)

    gains.sort(key=lambda x: (x["current_value"] or 0), reverse=True)
    return jsonify({"gains": gains, "display_currency": currency})


@app.route("/api/portfolio/yearly_yield")
def portfolio_yearly_yield():
    """
    Portfolio yearly return metrics.

    Query params:
      currency=ILS|USD|EUR|GBP
      year=YYYY   (optional; if omitted returns all years)
    """
    eng = get_engine()
    currency = _currency()
    year_raw = request.args.get("year")

    if year_raw:
        try:
            year = int(year_raw)
        except ValueError:
            return jsonify({"error": "Invalid year parameter"}), 400
        return jsonify(eng.get_yearly_yield(year, currency))

    return jsonify(eng.get_all_yearly_yields(currency))


@app.route("/api/portfolio/yield_series")
def portfolio_yield_series():
    """
    Rolling yield series for charting.

    Query params:
      currency=ILS|USD|EUR|GBP
      frequency=W|M      (default: W)
      window_days=int    (default: 365)
    """
    eng = get_engine()
    currency = _currency()
    freq = request.args.get("frequency", "W").upper()
    if freq not in ("W", "M"):
        freq = "W"
    try:
        window_days = int(request.args.get("window_days", "365"))
    except ValueError:
        return jsonify({"error": "Invalid window_days parameter"}), 400

    return jsonify(eng.get_rolling_return_timeseries(
        display_currency=currency,
        point_frequency=freq,
        window_days=window_days,
    ))


def _build_cashflow(eng, cash_name: str, cash_currency: str):
    """
    Builds a cashflow event list for a cash asset (NIS=ILS or USD=USD).

    Handles:
      - Initial balance
      - Monthly fees in this currency
      - Deposits in this currency
      - Buys/sells of regular assets in this currency (−/+ amount, − fee)
      - FX exchanges INTO this currency (buy USD with ILS → credit USD)
      - FX exchanges OUT OF this currency (buy NIS with USD → debit USD)
      - Dividends from assets denominated in this currency
    """
    from datetime import date as date_cls
    from dateutil.relativedelta import relativedelta

    events = []

    # Initial balance
    initial_qty = eng.initial_holdings.get(cash_name, {}).get("quantity", 0.0)
    events.append({
        "date": eng.start_date.isoformat(),
        "type": "initial",
        "label": "Initial Balance",
        "amount": initial_qty,
        "symbol": None,
    })

    # Monthly fees
    cur_month = eng.start_date.replace(day=1)
    today = eng.end_date
    while cur_month <= today:
        cfg = eng.get_fee_config_at_date(cur_month)
        if cfg["monthly_fee_currency"] == cash_currency:
            events.append({
                "date": cur_month.isoformat(),
                "type": "monthly_fee",
                "label": "Monthly Fee",
                "amount": -cfg["monthly_fee"],
                "symbol": None,
            })
        cur_month += relativedelta(months=1)

    # Actions
    for action in eng.actions:
        atype    = action["action_type"]
        currency = action.get("currency") or "ILS"
        amount   = action.get("amount") or (
            (action.get("quantity") or 0) * (action.get("cost_per_unit") or 0))
        adate    = action["date"]
        adate_obj = date_cls.fromisoformat(adate)
        symbol   = action.get("symbol") or ""

        # Detect FX exchange: buying/selling a cash asset
        sym_info = eng.initial_holdings.get(symbol, {})
        is_cash_symbol = sym_info.get("asset_type") == "cash"

        if atype == "deposit" and currency == cash_currency and amount:
            events.append({"date": adate, "type": "deposit", "label": "Deposit",
                           "amount": amount, "symbol": symbol or None})

        elif atype == "buy" and amount:
            fee = eng._calc_transaction_fee(amount, currency, adate_obj)

            if is_cash_symbol and currency == cash_currency:
                # FX exchange: spending this currency to buy another cash
                # e.g. NIS cashflow: "buy USD" with ILS → NIS -= amount + fee
                dest_currency = sym_info.get("currency", symbol)
                fx = eng.get_rate_on_date(cash_currency, dest_currency, adate_obj)
                usd_received = round(amount * fx, 4)
                events.append({"date": adate, "type": "fx_exchange",
                                "label": f"FX Exchange → {symbol} ({usd_received:+.2f} {dest_currency})",
                                "amount": -(amount + fee), "symbol": symbol or None,
                                "fx_rate": round(fx, 4), "received": usd_received,
                                "received_currency": dest_currency})

            elif is_cash_symbol and sym_info.get("currency") == cash_currency and currency != cash_currency:
                # FX exchange: receiving this currency by spending another
                # e.g. USD cashflow: "buy USD" with ILS → USD += amount * fx
                fx = eng.get_rate_on_date(currency, cash_currency, adate_obj)
                usd_received = round(amount * fx, 4)
                events.append({"date": adate, "type": "fx_exchange",
                                "label": f"FX Exchange ← {currency} ({amount:.2f} {currency})",
                                "amount": usd_received, "symbol": symbol or None,
                                "fx_rate": round(fx, 4)})

            elif currency == cash_currency:
                # Normal buy paid in this currency
                events.append({"date": adate, "type": "buy", "label": f"Buy {symbol}",
                                "amount": -amount, "symbol": symbol or None})
                events.append({"date": adate, "type": "transaction_fee",
                                "label": f"Fee ({symbol})", "amount": -fee,
                                "symbol": symbol or None})

        elif atype == "sell" and amount:
            fee = eng._calc_transaction_fee(amount, currency, adate_obj)

            if is_cash_symbol and sym_info.get("currency") == cash_currency and currency != cash_currency:
                # FX exchange: selling another cash to receive this currency
                fx = eng.get_rate_on_date(currency, cash_currency, adate_obj)
                received = round(amount * fx, 4)
                events.append({"date": adate, "type": "fx_exchange",
                                "label": f"FX Exchange ← {currency} ({amount:.2f} {currency})",
                                "amount": received, "symbol": symbol or None,
                                "fx_rate": round(fx, 4)})

            elif currency == cash_currency:
                events.append({"date": adate, "type": "sell", "label": f"Sell {symbol}",
                                "amount": amount, "symbol": symbol or None})
                events.append({"date": adate, "type": "transaction_fee",
                                "label": f"Fee ({symbol})", "amount": -fee,
                                "symbol": symbol or None})

    # Dividends from assets in this currency
    all_assets: dict[str, str] = {}
    for name, info in eng.initial_holdings.items():
        if info.get("currency") == cash_currency and info.get("ticker") and info["ticker"] != "USDILS=X":
            all_assets[name] = info["ticker"]
    for action in eng.actions:
        if action["action_type"] == "buy" and action.get("currency") == cash_currency:
            sym = action.get("symbol")
            tkr = action.get("ticker")
            if sym and tkr and sym not in eng.initial_holdings and sym not in all_assets:
                all_assets[sym] = tkr

    for name, ticker in all_assets.items():
        ticker_currency = eng.get_ticker_currency(ticker)
        div_series = eng.fetch_dividend_history(ticker, eng.start_date, eng.end_date)
        for div_ts, div_per_share in div_series.items():
            div_date = div_ts.date() if hasattr(div_ts, "date") else div_ts
            cfg = eng.get_fee_config_at_date(div_date)
            holdings_snap = eng.get_holdings_at_date(div_date)
            qty = eng._resolve_quantity(name, ticker, holdings_snap)
            if qty <= 0:
                continue
            fx = eng.get_rate_on_date(ticker_currency, cash_currency, div_date)
            gross = qty * div_per_share * fx
            tax   = gross * cfg["dividend_tax_rate"]
            net   = gross - tax
            events.append({
                "date": div_date.isoformat(),
                "type": "dividend",
                "label": f"Dividend {name}",
                "amount": net,
                "symbol": name,
                "gross": round(gross, 4),
                "tax": round(tax, 4),
            })

    events.sort(key=lambda e: e["date"])

    # Accumulate running balance
    balance = 0.0
    cashflow = []
    for ev in events:
        balance += ev["amount"]
        ev["balance"] = round(balance, 4)
        ev["amount"]  = round(ev["amount"], 4)
        cashflow.append(ev)

    return cashflow, round(balance, 4)


@app.route("/api/asset/USD/cashflow")
def usd_cashflow():
    """Returns a detailed cash-flow breakdown for the USD cash asset."""
    eng = get_engine()
    cashflow, final_balance = _build_cashflow(eng, "USD", "USD")
    return jsonify({"cashflow": cashflow, "final_balance": final_balance, "currency": "USD"})


@app.route("/api/asset/NIS/cashflow")
def nis_cashflow():
    """Returns a detailed cash-flow breakdown for the NIS (ILS cash) asset."""
    eng = get_engine()
    cashflow, final_balance = _build_cashflow(eng, "NIS", "ILS")
    return jsonify({"cashflow": cashflow, "final_balance": final_balance, "currency": "ILS"})



@app.route("/api/asset/<path:name>/history")
def asset_history(name: str):
    """Single asset value history."""
    eng = get_engine()
    currency = _currency()
    frequency = _frequency()

    # Find holding info
    info = eng.initial_holdings.get(name)
    if not info:
        for action in eng.actions:
            if action.get("symbol") == name:
                info = {
                    "quantity": 0.0,
                    "ticker": action.get("ticker"),
                    "currency": action.get("currency") or "ILS",
                    "asset_type": "stock",
                }
                break
    if not info:
        return jsonify({"error": f"Asset '{name}' not found"}), 404

    dates = eng._date_range(eng.start_date, eng.end_date, frequency)
    result = eng.get_asset_value_history(name, info, currency, dates)
    return jsonify(result)


@app.route("/api/asset/<path:name>/upload_prices", methods=["POST"])
def upload_asset_prices(name: str):
    """
    Upload a TASE-style Hebrew CSV file to provide price history for an asset
    that has no yfinance ticker.

    CSV format (as exported from TASE):
      Header rows (2-3 lines of metadata, skipped)
      Column header:  תאריך, שער נעילה מתואם(באגורות), ...
      Data rows:      DD/MM/YYYY, price_in_agorot, ...

    Stores prices (divided by 100 to convert Agorot → ILS) into the engine cache
    under key  "price_MANUAL:<name>"  and updates the holding's ticker to this
    virtual key so the engine can serve prices from cache.

    Returns: { "rows_imported": N, "date_range": [first, last] }
    """
    eng = get_engine()

    # Find the holding — check initial_holdings first, then action-only assets
    info = eng.initial_holdings.get(name)
    action_only = False
    if not info:
        # Search buy actions for this symbol
        for action in eng.actions:
            if action["action_type"] == "buy" and action.get("symbol") == name:
                info = {
                    "quantity": 0.0,
                    "ticker": action.get("ticker"),
                    "currency": action.get("currency") or "ILS",
                    "asset_type": "stock",
                }
                action_only = True
                break
    if not info:
        return jsonify({"error": f"Asset '{name}' not found in holdings or actions"}), 404

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded (field name: 'file')"}), 400

    f = request.files["file"]
    raw = f.read().decode("utf-8-sig", errors="replace")  # strip BOM

    rows_imported = 0
    price_data: dict[str, float] = {}

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue

        # ── Format 1: TASE ETF export ──────────────────────────────────────
        # Columns: תאריך (DD/MM/YYYY), שער נעילה מתואם(באגורות), ...
        # Date uses forward-slash separators
        if re.match(r"\d{2}/\d{2}/\d{4}$", parts[0]):
            try:
                d = datetime.strptime(parts[0], "%d/%m/%Y").date()
                price_ils = float(parts[1]) / 100.0   # Agorot → ILS
                price_data[d.isoformat()] = price_ils
                rows_imported += 1
            except (ValueError, IndexError):
                pass
            continue

        # ── Format 2: Mutual Fund history (e.g. gemel.co.il) ───────────────
        # Columns: מס' קרן, תאריך (DD.MM.YYYY), מחיר קניה (Agorot), מחיר פדיון, ...
        # First column is numeric fund ID; date uses dot separators
        if len(parts) >= 3 and re.match(r"\d+$", parts[0]) and re.match(r"\d{2}\.\d{2}\.\d{4}$", parts[1]):
            try:
                d = datetime.strptime(parts[1], "%d.%m.%Y").date()
                price_ils = float(parts[2]) / 100.0   # Agorot → ILS
                price_data[d.isoformat()] = price_ils
                rows_imported += 1
            except (ValueError, IndexError):
                pass
            continue
        # Skip header rows and unrecognised lines silently

    if not price_data:
        return jsonify({"error": "No valid price rows found in CSV"}), 400

    # Use a virtual ticker key: "CSV:<name>"
    virtual_ticker = f"CSV:{name}"
    cache_key = f"price_{virtual_ticker}"

    # ── Write all cache entries atomically in one save ───────────────────
    eng._cache.setdefault(cache_key, {}).update(price_data)
    # Currency metadata: already ILS (divided during parse, not ILA)
    eng._cache[f"raw_currency_{virtual_ticker}"] = {"value": "ILS"}
    eng._cache[f"currency_{virtual_ticker}"]     = {"value": "ILS"}
    eng._save_cache()   # single write — all three entries flushed to disk

    # ── Patch portfolio.json with the virtual ticker ─────────────────────
    portfolio = json.loads(PORTFOLIO_JSON.read_text(encoding="utf-8"))
    if name in portfolio["initial_holdings"]:
        # Existing initial holding — just update ticker + note
        portfolio["initial_holdings"][name]["ticker"] = virtual_ticker
        old_note = portfolio["initial_holdings"][name].get("note") or ""
        old_note = re.sub(r"\s*\[CSV prices loaded:[^\]]*\]", "", old_note).strip()
        portfolio["initial_holdings"][name]["note"] = (
            f"{old_note} [CSV prices loaded: {rows_imported} rows]".strip()
        )
    elif action_only:
        # Action-only asset — promote it into initial_holdings with qty=0
        # (actual qty comes from buy actions replay)
        portfolio["initial_holdings"][name] = {
            "quantity": 0.0,
            "ticker": virtual_ticker,
            "currency": info.get("currency") or "ILS",
            "asset_type": info.get("asset_type", "stock"),
            "note": f"Promoted from action — CSV prices loaded: {rows_imported} rows",
        }
    PORTFOLIO_JSON.write_text(
        json.dumps(portfolio, indent=2, ensure_ascii=False), encoding="utf-8")

    # Reload engine — it will read the updated portfolio.json and price cache
    _reload_engine()

    sorted_dates = sorted(price_data.keys())
    return jsonify({
        "asset": name,
        "virtual_ticker": virtual_ticker,
        "rows_imported": rows_imported,
        "date_range": [sorted_dates[0], sorted_dates[-1]],
        "sample_price": price_data[sorted_dates[-1]],
    })


@app.route("/api/asset/<path:name>/clear_prices", methods=["POST"])
def clear_asset_prices(name: str):
    """Remove CSV-loaded prices and reset ticker to null for an asset."""
    eng = get_engine()
    portfolio = json.loads(PORTFOLIO_JSON.read_text(encoding="utf-8"))
    if name not in portfolio["initial_holdings"]:
        return jsonify({"error": f"Asset '{name}' not found"}), 404

    info = portfolio["initial_holdings"][name]
    virtual_ticker = info.get("ticker", "")
    if virtual_ticker and virtual_ticker.startswith("CSV:"):
        # Remove from cache
        cache_key = f"price_{virtual_ticker}"
        eng._cache.pop(cache_key, None)
        eng._cache.pop(f"raw_currency_{virtual_ticker}", None)
        eng._cache.pop(f"currency_{virtual_ticker}", None)
        eng._save_cache()
        # If this was a promoted action-only asset, remove entirely from initial_holdings
        was_promoted = "Promoted from action" in (info.get("note") or "")
        if was_promoted:
            del portfolio["initial_holdings"][name]
        else:
            info["ticker"] = None
            note = info.get("note", "")
            note = re.sub(r"\s*\[CSV prices loaded:[^\]]*\]", "", note).strip()
            info["note"] = note or None
        PORTFOLIO_JSON.write_text(
            json.dumps(portfolio, indent=2, ensure_ascii=False), encoding="utf-8")
        _reload_engine()
        return jsonify({"cleared": True, "asset": name})
    return jsonify({"error": "Asset does not have CSV prices loaded"}), 400


# ── Action CRUD ────────────────────────────────────────────────────────────

def _autofill_action(data: dict) -> dict:
    """
    Fill in missing numeric fields from the other two:
      amount = quantity × cost_per_unit   (if amount missing)
      cost_per_unit = amount / quantity   (if cpu missing)
      quantity = amount / cost_per_unit   (if qty missing)
    All values are rounded to 4 decimal places.
    """
    qty    = data.get("quantity")
    cpu    = data.get("cost_per_unit")
    amount = data.get("amount")

    if qty and cpu and not amount:
        data["amount"] = round(qty * cpu, 4)
    elif qty and amount and not cpu:
        data["cost_per_unit"] = round(amount / qty, 4)
    elif cpu and amount and not qty:
        data["quantity"] = round(amount / cpu, 4)

    return data


@app.route("/api/actions", methods=["GET"])
def get_actions():
    eng = get_engine()
    return jsonify(eng.actions)


@app.route("/api/splits", methods=["GET"])
def get_splits():
    eng = get_engine()
    return jsonify(eng.get_splits())


@app.route("/api/actions", methods=["POST"])
def add_action():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    required = ["date", "action_type", "currency"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Load, mutate, save
    portfolio = json.loads(PORTFOLIO_JSON.read_text(encoding="utf-8"))
    new_action = {
        "date":           data["date"],
        "action_type":    data["action_type"],
        "symbol":         data.get("symbol"),
        "ticker":         data.get("ticker"),
        "quantity":       data.get("quantity"),
        "cost_per_unit":  data.get("cost_per_unit"),
        "amount":         data.get("amount"),
        "split_ratio":    data.get("split_ratio"),
        "currency":       data["currency"],
        "note":           data.get("note"),
    }
    _autofill_action(new_action)
    portfolio["actions"].append(new_action)
    portfolio["actions"].sort(key=lambda a: a["date"])
    PORTFOLIO_JSON.write_text(
        json.dumps(portfolio, indent=2, ensure_ascii=False), encoding="utf-8")
    _reload_engine()
    return jsonify(new_action), 201


@app.route("/api/actions/<int:idx>", methods=["PUT"])
def update_action(idx: int):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    portfolio = json.loads(PORTFOLIO_JSON.read_text(encoding="utf-8"))
    actions = portfolio["actions"]
    if idx < 0 or idx >= len(actions):
        return jsonify({"error": "Index out of range"}), 404

    actions[idx].update(data)
    _autofill_action(actions[idx])
    portfolio["actions"].sort(key=lambda a: a["date"])
    PORTFOLIO_JSON.write_text(
        json.dumps(portfolio, indent=2, ensure_ascii=False), encoding="utf-8")
    _reload_engine()
    return jsonify(actions[idx])


@app.route("/api/actions/<int:idx>", methods=["DELETE"])
def delete_action(idx: int):
    portfolio = json.loads(PORTFOLIO_JSON.read_text(encoding="utf-8"))
    actions = portfolio["actions"]
    if idx < 0 or idx >= len(actions):
        return jsonify({"error": "Index out of range"}), 404

    deleted = actions.pop(idx)
    PORTFOLIO_JSON.write_text(
        json.dumps(portfolio, indent=2, ensure_ascii=False), encoding="utf-8")
    _reload_engine()
    return jsonify({"deleted": deleted})


# ── Startup ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    get_engine()          # load + pre-warm on startup
    app.run(debug=False, host="0.0.0.0", port=5000)

