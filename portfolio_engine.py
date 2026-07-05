"""
Portfolio Engine
================
Loads portfolio.json + fees.json, fetches price/dividend/exchange-rate
history from yfinance (with disk cache), and computes:

  - Holdings at any date
  - Per-asset value history (price + net dividends after tax)
  - Per-asset gain summary (unrealized gain + dividends - fees)
  - Full portfolio timeline (sum of all assets)
  - All fees & taxes (monthly, transaction, capital-gains)

All monetary results are returned in the caller-chosen display_currency.
"""

import json
import logging
import math
import re
from html import unescape
from io import StringIO
from bisect import bisect_right
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
PORTFOLIO_JSON = DATA_DIR / "portfolio.json"
FEES_JSON      = DATA_DIR / "fees.json"
CACHE_JSON     = DATA_DIR / "price_cache.json"

# Supported display currencies and their yfinance forex ticker to ILS
FOREX_TICKERS = {
    ("USD", "ILS"): "USDILS=X",
    ("ILS", "USD"): "ILSUSD=X",
    ("GBP", "ILS"): "GBPILS=X",
    ("ILS", "GBP"): "ILSGBP=X",
    ("EUR", "ILS"): "EURILS=X",
    ("ILS", "EUR"): "ILSEUR=X",
    ("USD", "EUR"): "USDEUR=X",
    ("EUR", "USD"): "EURUSD=X",
    ("USD", "GBP"): "USDGBP=X",
    ("GBP", "USD"): "GBPUSD=X",
    ("EUR", "GBP"): "EURGBP=X",
    ("GBP", "EUR"): "GBPEUR=X",
}


# ─────────────────────────────────────────────────────────────────────────────
class PortfolioEngine:

    _CSV_AUTOFETCH_VERSION = 3
    _CSV_AUTOFETCH_OK_COOLDOWN_DAYS = 1
    _CSV_AUTOFETCH_FAIL_COOLDOWN_DAYS = 1
    _DENSE_SERIES_STALE_DAYS = 2
    _DIVIDEND_REFRESH_COOLDOWN_DAYS = 1

    def __init__(self,
                 portfolio_path: Path = PORTFOLIO_JSON,
                 fees_path: Path = FEES_JSON,
                 cache_path: Path = CACHE_JSON):

        with open(portfolio_path, encoding="utf-8") as f:
            pdata = json.load(f)
        self.initial_holdings: dict = pdata["initial_holdings"]
        self.actions: list[dict]   = sorted(pdata["actions"], key=lambda a: a["date"])

        with open(fees_path, encoding="utf-8") as f:
            self.fees_config: dict = json.load(f)

        self._cache_path = cache_path
        self._cache: dict = self._load_cache()

        # derive portfolio date range
        self.start_date = date.fromisoformat(pdata.get("start_date") or self.actions[0]["date"])
        self.end_date   = date.today()

        # ── In-memory memoization (not persisted — cleared on reload) ────────
        self._holdings_memo: dict[str, dict] = {}   # date_iso → holdings dict
        self._fee_config_memo: dict[str, dict] = {} # date_iso → fee config dict
        self._rate_memo: dict[str, float] = {}      # "FROM_TO_date" → rate
        self._result_memo: dict[tuple, dict] = {}    # heavy computed API payloads

    # ─────────────────────────────────────────────────────────────────────────
    # CACHE HELPERS
    #
    # Schema per key (e.g. "price_AAPL", "fx_USDILS=X", "div_AAPL"):
    #   { "YYYY-MM-DD": value, ... }   ← flat date→value dict, all known dates
    #
    # Strategy:
    #   - Gap before stored window  → fetch [start, cache_start-1]
    #   - Gap after stored window   → fetch [cache_end+1, end]
    #   - Today stale               → if cache_end < today, re-fetch [cache_end, end]
    #     (detected purely from the latest date in the series, no extra field needed)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict:
        if self._cache_path.exists():
            try:
                return json.loads(self._cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_cache(self):
        self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        self._cache_dirty = False

    def _mark_cache_dirty(self):
        """Mark cache as needing a save. Call flush_cache() when done with a batch."""
        self._cache_dirty = True

    def flush_cache(self):
        """Write cache to disk if there are pending changes."""
        if getattr(self, "_cache_dirty", False):
            self._save_cache()

    def _series_cache_set(self, key: str, new_data: dict):
        """Merges new date→value pairs into the flat stored dict for key."""
        self._cache.setdefault(key, {}).update(new_data)
        self._mark_cache_dirty()

    def _meta_cache_set(self, key: str, value):
        self._cache[key] = {"value": value}
        self._mark_cache_dirty()

    def _series_cache_get(self, key: str, start: date, end: date
                          ) -> tuple[pd.Series | None, list[tuple[date, date]]]:
        """
        Returns (cached_series_for_range, missing_ranges).
        end is clamped to today — we never request future data from yfinance.
        """
        today   = date.today()
        end     = min(end, today)          # clamp: no future fetches
        stored: dict = self._cache.get(key, {})

        if not stored:
            return None, [(start, end)]

        all_dates   = sorted(stored.keys())
        cache_start = date.fromisoformat(all_dates[0])
        cache_end   = date.fromisoformat(all_dates[-1])

        missing: list[tuple[date, date]] = []

        # Gap before cached window
        if start < cache_start:
            missing.append((start, cache_start - timedelta(days=1)))

        # Gap after cached window (also covers stale today)
        # For dense series (prices/fx), skip tiny trailing gaps to keep requests cache-first.
        trailing_start = cache_end + timedelta(days=1)
        if cache_end < today:
            cache_age = (today - cache_end).days
            if cache_age > self._DENSE_SERIES_STALE_DAYS:
                trailing_start = cache_end
            else:
                trailing_start = today + timedelta(days=1)
        if end >= trailing_start:
            fetch_from = max(trailing_start, start)
            if fetch_from <= end:
                missing.append((fetch_from, end))

        result = {d: v for d, v in stored.items()
                  if start.isoformat() <= d <= end.isoformat()}
        s = pd.Series(result) if result else None
        if s is not None:
            s.index = pd.to_datetime(s.index)
        return s, missing

    # Kept for scalar metadata (e.g. ticker currency) which is not a time series
    def _meta_cache_get(self, key: str):
        entry = self._cache.get(key)
        if not entry or "value" not in entry:
            return None
        return entry["value"]

    # ─────────────────────────────────────────────────────────────────────────
    # FEE CONFIG RESOLVER
    # ─────────────────────────────────────────────────────────────────────────

    def get_fee_config_at_date(self, d: date) -> dict:
        """
        Returns the applicable fee rules for a given date by finding the
        last entry in each fee list whose valid_from <= d.
        Results are memoized in memory.
        """
        ds = d.isoformat()
        if ds in self._fee_config_memo:
            return self._fee_config_memo[ds]

        def resolve(entries: list) -> dict:
            chosen = entries[0]
            for e in entries:
                if e["valid_from"] <= ds:
                    chosen = e
            return chosen

        monthly   = resolve(self.fees_config["monthly_fee"])
        ils_fee   = resolve(self.fees_config["transaction_fee"]["ILS"])
        usd_fee   = resolve(self.fees_config["transaction_fee"]["USD"])
        gbp_fee   = resolve(self.fees_config["transaction_fee"].get("GBP", [{"valid_from":"2000-01-01","rate":0.002,"minimum":2.5,"currency":"GBP"}]))
        eur_fee   = resolve(self.fees_config["transaction_fee"].get("EUR", [{"valid_from":"2000-01-01","rate":0.002,"minimum":3.0,"currency":"EUR"}]))
        div_tax   = resolve(self.fees_config["dividend_tax_rate"])
        cg_tax    = resolve(self.fees_config["capital_gains_tax_rate"])

        result = {
            "monthly_fee":           monthly["amount"],
            "monthly_fee_currency":  monthly["currency"],
            "ILS_rate":              ils_fee["rate"],
            "ILS_minimum":           ils_fee["minimum"],
            "USD_rate":              usd_fee["rate"],
            "USD_minimum":           usd_fee["minimum"],
            "GBP_rate":              gbp_fee["rate"],
            "GBP_minimum":           gbp_fee["minimum"],
            "EUR_rate":              eur_fee["rate"],
            "EUR_minimum":           eur_fee["minimum"],
            "dividend_tax_rate":     div_tax["rate"],
            "capital_gains_tax_rate": cg_tax["rate"],
        }
        self._fee_config_memo[ds] = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # EXCHANGE RATE
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_exchange_rate_history(self,
                                    from_currency: str,
                                    to_currency: str,
                                    start: date,
                                    end: date) -> pd.Series:
        """
        Returns a daily Series of exchange rates from_currency → to_currency.
        Same currency → returns Series of 1.0.
        Cache key = "fx_{ticker}"; stores a single growing time series.
        Only fetches missing date ranges.
        """
        from_currency = from_currency or "ILS"
        to_currency   = to_currency   or "ILS"
        if from_currency == to_currency:
            idx = pd.date_range(start, end, freq="D")
            return pd.Series(1.0, index=idx)

        pair   = (from_currency, to_currency)
        ticker = FOREX_TICKERS.get(pair)
        if not ticker:
            inv = FOREX_TICKERS.get((to_currency, from_currency))
            if inv:
                s = self.fetch_exchange_rate_history(to_currency, from_currency, start, end)
                return (1.0 / s).replace([float("inf")], None).ffill()
            logger.warning("No forex ticker for %s→%s, defaulting to 1.0", from_currency, to_currency)
            return pd.Series(1.0, index=pd.date_range(start, end, freq="D"))

        cache_key = f"fx_{ticker}"
        cached_s, missing = self._series_cache_get(cache_key, start, end)

        for gap_start, gap_end in missing:
            try:
                raw = yf.Ticker(ticker).history(
                    start=gap_start.isoformat(),
                    end=(gap_end + timedelta(days=1)).isoformat())
                if not raw.empty:
                    s = raw["Close"].ffill().bfill()
                    s.index = s.index.tz_localize(None) if s.index.tz else s.index
                    self._series_cache_set(cache_key, {str(k.date()): v for k, v in s.items()})
            except Exception as e:
                logger.error("Failed to fetch FX %s [%s-%s]: %s", ticker, gap_start, gap_end, e)

        if missing:
            self.flush_cache()

        # Re-read full range from cache after filling gaps
        stored = self._cache.get(cache_key, {})
        result = {d: v for d, v in stored.items()
                  if start.isoformat() <= d <= end.isoformat()
                  and isinstance(v, (int, float))}  # skip metadata entries

        # If direct data is missing or incomplete, try the inverse pair from cache
        inv_pair = (to_currency, from_currency)
        inv_ticker = FOREX_TICKERS.get(inv_pair)
        if inv_ticker and inv_ticker != ticker:
            inv_cache_key = f"fx_{inv_ticker}"
            inv_stored = self._cache.get(inv_cache_key, {})
            inv_result = {d: v for d, v in inv_stored.items()
                          if start.isoformat() <= d <= end.isoformat()
                          and isinstance(v, (int, float)) and v != 0}
            if inv_result:
                # Merge: prefer direct data, fill gaps with 1/inverse
                for d_str, inv_v in inv_result.items():
                    if d_str not in result:
                        result[d_str] = 1.0 / inv_v

        if not result:
            return pd.Series(1.0, index=pd.date_range(start, end, freq="D"))
        s = pd.Series(result)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.sort_index()

    def get_rate_on_date(self, from_currency: str, to_currency: str, d: date) -> float:
        """Single-date exchange rate lookup with ±5 day fallback window. Memoized."""
        from_currency = from_currency or "ILS"
        to_currency   = to_currency   or "ILS"
        if from_currency == to_currency:
            return 1.0
        memo_key = f"{from_currency}_{to_currency}_{d.isoformat()}"
        if memo_key in self._rate_memo:
            return self._rate_memo[memo_key]
        s = self.fetch_exchange_rate_history(from_currency, to_currency,
                                             d - timedelta(days=7), d + timedelta(days=1))
        s.index = pd.to_datetime(s.index).tz_localize(None)
        target = pd.Timestamp(d)
        if target in s.index:
            rate = float(s[target])
        else:
            past = s[s.index <= target]
            rate = float(past.iloc[-1]) if not past.empty else (float(s.iloc[0]) if not s.empty else 1.0)
        self._rate_memo[memo_key] = rate
        return rate

    # ─────────────────────────────────────────────────────────────────────────
    # TICKER NATIVE CURRENCY
    # ─────────────────────────────────────────────────────────────────────────

    def get_ticker_currency(self, ticker: str) -> str:
        """Returns the native trading currency of a ticker (cached).
        ILA (Israeli Agorot) is normalised to ILS — prices are divided by 100
        in fetch_price_history and fetch_dividend_history.
        CSV virtual tickers (prefix 'CSV:') always return ILS.
        """
        if ticker and ticker.startswith("CSV:"):
            return "ILS"
        cached = self._meta_cache_get(f"currency_{ticker}")
        if cached:
            return cached
        try:
            info = yf.Ticker(ticker).fast_info
            raw_cur = getattr(info, "currency", None) or "USD"
            # Cache the raw currency too (avoids a second API call in _is_agorot_ticker)
            self._meta_cache_set(f"raw_currency_{ticker}", raw_cur)
            # ILA = Israeli Agorot (1/100 ILS) — normalise to ILS
            cur = "ILS" if raw_cur == "ILA" else raw_cur
            self._meta_cache_set(f"currency_{ticker}", cur)
            self.flush_cache()
            return cur
        except Exception:
            # Cache the fallback so we don't retry every time
            self._meta_cache_set(f"currency_{ticker}", "USD")
            self._meta_cache_set(f"raw_currency_{ticker}", "USD")
            self.flush_cache()
            return "USD"

    # Tickers whose prices are quoted in Agorot (ILA) and need ÷100
    def _is_agorot_ticker(self, ticker: str) -> bool:
        """Returns True if yfinance reports this ticker's currency as ILA.
        CSV virtual tickers are pre-converted to ILS during upload — always False.
        """
        if ticker and ticker.startswith("CSV:"):
            return False
        cached = self._meta_cache_get(f"raw_currency_{ticker}")
        if cached is not None:
            return cached == "ILA"
        try:
            info = yf.Ticker(ticker).fast_info
            cur  = getattr(info, "currency", None) or "USD"
            self._meta_cache_set(f"raw_currency_{ticker}", cur)
            # Also cache the normalized currency (avoids a second API call in get_ticker_currency)
            self._meta_cache_set(f"currency_{ticker}", "ILS" if cur == "ILA" else cur)
            self.flush_cache()
            return cur == "ILA"
        except Exception:
            # Cache the fallback so we don't retry every time
            self._meta_cache_set(f"raw_currency_{ticker}", "USD")
            self.flush_cache()
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # PRICE HISTORY
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _http_get_text(url: str, timeout: int = 30) -> str:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")

    @staticmethod
    def _http_post_json(url: str, payload: dict, timeout: int = 30, referer: str | None = None) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
        }
        if referer:
            headers["Referer"] = referer
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")

    @staticmethod
    def _pick_tase_table_columns(columns: list[str]) -> tuple[str | None, str | None]:
        date_col = None
        price_col = None
        for col in columns:
            c = col.strip().lower()
            if date_col is None and ("תאריך" in col or "date" in c):
                date_col = col
            if price_col is None and (
                "שער" in col or "מחיר" in col or "נעילה" in col
                or "price" in c or "nav" in c
            ):
                price_col = col
        return date_col, price_col

    def _parse_tase_prices_from_html(self, html: str) -> dict[str, float]:
        try:
            tables = pd.read_html(StringIO(html))
        except Exception:
            tables = []

        best: dict[str, float] = {}
        for df in tables:
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip()
                              for tup in df.columns.values]
            cols = [str(c).strip() for c in df.columns]
            date_col, price_col = self._pick_tase_table_columns(cols)
            if not date_col or not price_col:
                continue

            dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
            prices = pd.to_numeric(
                df[price_col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
                .str.replace(r"[^0-9.\-]", "", regex=True),
                errors="coerce",
            )
            parsed = pd.DataFrame({"date": dates, "price": prices}).dropna()
            if parsed.empty:
                continue

            # TASE historical values are often published in Agorot.
            if parsed["price"].median() > 1000:
                parsed["price"] = parsed["price"] / 100.0

            series_dict = {
                d.date().isoformat(): float(p)
                for d, p in zip(parsed["date"], parsed["price"])
            }
            if len(series_dict) > len(best):
                best = series_dict

        if best:
            return best

        # Fallback: parse plain HTML table rows without extra dependencies.
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL)
        parsed_rows: list[list[str]] = []
        for row in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)
            clean_cells = [re.sub(r"<[^>]+>", "", unescape(c)).strip() for c in cells]
            if clean_cells:
                parsed_rows.append(clean_cells)

        fallback: dict[str, float] = {}
        for cells in parsed_rows:
            date_val = None
            price_val = None
            for i, token in enumerate(cells):
                t = token.strip()
                if date_val is None and re.match(r"\d{2}[./]\d{2}[./]\d{4}$", t):
                    date_val = t.replace(".", "/")
                    # Common format: date in col 0, adjusted close in col 1.
                    if i + 1 < len(cells):
                        raw_price = re.sub(r"[^0-9.\-]", "", cells[i + 1].replace(",", ""))
                        try:
                            price_val = float(raw_price)
                        except Exception:
                            price_val = None
                    break
            if date_val is None or price_val is None:
                continue
            try:
                d = pd.to_datetime(date_val, dayfirst=True, errors="coerce")
                if pd.isna(d):
                    continue
                fallback[d.date().isoformat()] = price_val
            except Exception:
                continue

        if fallback:
            vals = list(fallback.values())
            if vals and pd.Series(vals).median() > 1000:
                fallback = {k: v / 100.0 for k, v in fallback.items()}
        return fallback

    @staticmethod
    def _normalize_tase_series(series: dict[str, float]) -> dict[str, float]:
        if not series:
            return {}
        vals = list(series.values())
        if vals and pd.Series(vals).median() > 1000:
            return {k: v / 100.0 for k, v in series.items()}
        return series

    def _parse_tase_graph_payload(self, payload: str) -> dict[str, float]:
        parsed: dict[str, float] = {}

        # JSON shape used by TASE chart API: {"history":[{"tdt":"DD/MM/YYYY","pval":1234.0}, ...]}
        try:
            obj = json.loads(payload)
            if isinstance(obj, dict):
                history = obj.get("history")
                if isinstance(history, list):
                    for row in history:
                        if not isinstance(row, dict):
                            continue
                        ds = row.get("tdt") or row.get("date") or row.get("tradeDate")
                        pv = row.get("pval")
                        if pv is None:
                            pv = row.get("price") or row.get("close") or row.get("value") or row.get("nav")
                        if ds is None or pv is None:
                            continue
                        dts = pd.to_datetime(ds, dayfirst=True, errors="coerce")
                        if pd.isna(dts):
                            continue
                        try:
                            parsed[dts.date().isoformat()] = float(pv)
                        except Exception:
                            continue
        except Exception:
            pass

        # Pattern A: [timestamp_ms, value]
        for ts_s, v_s in re.findall(r"\[\s*(\d{12,13})\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", payload):
            try:
                d = pd.to_datetime(int(ts_s), unit="ms", utc=True).tz_localize(None).date().isoformat()
                parsed[d] = float(v_s)
            except Exception:
                continue

        # Pattern B: objects with date + value fields
        patterns = [
            r'"(?:date|tradeDate|Date|xDate)"\s*:\s*"([^\"]+)"[^{}]{0,180}?"(?:price|close|value|nav|y|שער|נעילה)"\s*:\s*(-?\d+(?:\.\d+)?)',
            r'"(?:price|close|value|nav|y|שער|נעילה)"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"(?:date|tradeDate|Date|xDate)"\s*:\s*"([^\"]+)"',
        ]
        for idx, pat in enumerate(patterns):
            for m in re.findall(pat, payload):
                ds, vs = (m[0], m[1]) if idx == 0 else (m[1], m[0])
                try:
                    dts = pd.to_datetime(ds, dayfirst=True, errors="coerce")
                    if pd.isna(dts):
                        continue
                    parsed[dts.date().isoformat()] = float(vs)
                except Exception:
                    continue

        return self._normalize_tase_series(parsed)

    def _parse_tase_graph_prices_from_html(self, html: str) -> dict[str, float]:
        parsed = self._parse_tase_graph_payload(html)
        if parsed:
            return parsed

        scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.IGNORECASE | re.DOTALL)
        best: dict[str, float] = {}
        for block in scripts:
            cand = self._parse_tase_graph_payload(block)
            if len(cand) > len(best):
                best = cand
        return best

    def _parse_tase_graph_prices_with_playwright(self, url: str) -> dict[str, float]:
        """Optional browser fallback for JS/bot-protected graph pages."""
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            return {}

        candidates: list[dict[str, float]] = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def on_response(resp):
                try:
                    rurl = resp.url.lower()
                    # Use only the canonical chart API to avoid mixing unrelated JSON payloads.
                    if "/api/charts/gethistorydata" not in rurl:
                        return
                    text = resp.text()
                    cand = self._parse_tase_graph_payload(text)
                    if cand:
                        candidates.append(cand)
                except Exception:
                    return

            page.on("response", on_response)
            page.goto(url, wait_until="networkidle", timeout=90000)
            browser.close()

        best: dict[str, float] = {}
        for cand in candidates:
            if len(cand) > len(best):
                best = cand
        return best

    def _parse_maya_mutual_history_payload(self, payload: str) -> dict[str, float]:
        try:
            obj = json.loads(payload)
        except Exception:
            return {}
        if not isinstance(obj, list):
            return {}

        parsed: dict[str, float] = {}
        for row in obj:
            if not isinstance(row, dict):
                continue
            ds = row.get("tradeDate") or row.get("date")
            pv = row.get("sellPrice")
            if pv is None:
                pv = row.get("purchasePrice")
            if pv is None:
                pv = row.get("price") or row.get("close") or row.get("value")
            if ds is None or pv is None:
                continue
            try:
                dts = pd.to_datetime(ds, errors="coerce")
                if pd.isna(dts):
                    continue
                parsed[dts.date().isoformat()] = float(pv)
            except Exception:
                continue
        return self._normalize_tase_series(parsed)

    def _fetch_maya_mutual_history_from_api(self, source_url: str) -> dict[str, float]:
        m = re.search(r"/mutual-funds/(\d{7,8})/historical-data", source_url)
        if not m:
            return {}
        fund_id = m.group(1).lstrip("0") or m.group(1)
        api_url = f"https://maya.tase.co.il/api/v1/funds/mutual/{fund_id}/history"

        merged: dict[str, float] = {}
        page_size = 20
        for page in range(1, 500):
            text = self._http_post_json(
                api_url,
                {"pageSize": page_size, "pageNumber": page, "period": 3},
                referer=source_url,
            )
            chunk = self._parse_maya_mutual_history_payload(text)
            if not chunk:
                break
            merged.update(chunk)
            if len(chunk) < page_size:
                break
        return merged

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        return [u.rstrip(")]},.") for u in re.findall(r"https?://[^\s\"'<>]+", text or "")]

    def _find_tase_historical_url_for_csv_ticker(self, ticker: str) -> str | None:
        if ticker.startswith("CSV:https://") or ticker.startswith("CSV:http://"):
            return ticker[4:]

        source_texts: list[str] = []
        csv_symbol = ticker[4:] if ticker.startswith("CSV:") else ticker
        for name, info in self.initial_holdings.items():
            holding_ticker = info.get("ticker")
            if holding_ticker == ticker or holding_ticker == csv_symbol:
                source_texts.append(str(info.get("note") or ""))
                source_texts.append(name)

        # Some promoted CSV assets keep the TASE URL only on the originating buy action.
        for action in self.actions:
            action_ticker = action.get("ticker")
            action_symbol = action.get("symbol")
            if (
                action_ticker in {ticker, csv_symbol}
                or action_symbol in {ticker, csv_symbol}
            ):
                source_texts.append(str(action.get("note") or ""))
                if action_symbol:
                    source_texts.append(str(action_symbol))

        for txt in source_texts:
            for url in self._extract_urls(txt):
                if (
                    ("maya.tase.co.il" in url and "historical-data" in url)
                    or ("market.tase.co.il" in url and "/graph" in url)
                ):
                    return url

        for txt in source_texts:
            m_id = re.search(r"\b(\d{7})\.TA\b", txt, flags=re.IGNORECASE)
            if not m_id:
                m_id = re.search(r"\b(\d{7})\b", txt)
            if m_id:
                fund_id = m_id.group(1)
                return f"https://maya.tase.co.il/he/funds/mutual-funds/{fund_id}/historical-data?period=3"
        return None

    def _is_csv_autofetch_due(self, ticker: str) -> bool:
        marker_key = f"csv_autofetch_status_{ticker}"
        marker = self._meta_cache_get(marker_key)
        if not isinstance(marker, dict):
            return True
        if marker.get("version") != self._CSV_AUTOFETCH_VERSION:
            return True

        marker_date = marker.get("date")
        if not marker_date:
            return True
        try:
            marker_day = date.fromisoformat(marker_date)
        except (TypeError, ValueError):
            return True

        age_days = (date.today() - marker_day).days
        status = marker.get("status")
        if status == "ok":
            return age_days >= self._CSV_AUTOFETCH_OK_COOLDOWN_DAYS
        if status == "failed":
            return age_days >= self._CSV_AUTOFETCH_FAIL_COOLDOWN_DAYS
        return True

    def _is_dividend_refresh_due(self, ticker: str) -> bool:
        marker = self._meta_cache_get(f"div_refresh_status_{ticker}")
        if not isinstance(marker, dict):
            return True
        marker_date = marker.get("date")
        if not marker_date:
            return True
        try:
            marker_day = date.fromisoformat(marker_date)
        except (TypeError, ValueError):
            return True
        return (date.today() - marker_day).days >= self._DIVIDEND_REFRESH_COOLDOWN_DAYS

    def _maybe_autofetch_csv_prices(self, ticker: str) -> bool:
        if not self._is_csv_autofetch_due(ticker):
            return False

        marker_key = f"csv_autofetch_status_{ticker}"
        today_iso = date.today().isoformat()

        source_url = self._find_tase_historical_url_for_csv_ticker(ticker)
        if not source_url:
            return False

        try:
            price_data: dict[str, float] = {}
            source_kind = "unknown"
            if "maya.tase.co.il" in source_url and "historical-data" in source_url:
                # Prefer Maya's JSON API over brittle HTML scraping for mutual funds.
                price_data = self._fetch_maya_mutual_history_from_api(source_url)
                if price_data:
                    source_kind = "maya_api"

            if not price_data:
                html = self._http_get_text(source_url)
                if "market.tase.co.il" in source_url and "/graph" in source_url:
                    price_data = self._parse_tase_graph_prices_from_html(html)
                    if price_data:
                        source_kind = "market_graph_html"
                    if not price_data:
                        price_data = self._parse_tase_graph_prices_with_playwright(source_url)
                        if price_data:
                            source_kind = "market_graph_playwright"
                elif "maya.tase.co.il" in source_url and "historical-data" in source_url:
                    price_data = self._parse_tase_prices_from_html(html)
                    if price_data:
                        source_kind = "maya_historical_html"
                    if not price_data:
                        price_data = self._parse_tase_graph_prices_with_playwright(source_url)
                        if price_data:
                            source_kind = "maya_historical_playwright"
                else:
                    price_data = self._parse_tase_prices_from_html(html)
                    if price_data:
                        source_kind = "generic_html"
            if not price_data:
                self._meta_cache_set(marker_key, {
                    "status": "failed",
                    "date": today_iso,
                    "version": self._CSV_AUTOFETCH_VERSION,
                })
                self.flush_cache()
                return False

            cache_key = f"price_{ticker}"
            self._series_cache_set(cache_key, price_data)
            self._meta_cache_set(f"raw_currency_{ticker}", "ILS")
            self._meta_cache_set(f"currency_{ticker}", "ILS")
            self._meta_cache_set(marker_key, {
                "status": "ok",
                "date": today_iso,
                "rows": len(price_data),
                "source": source_kind,
                "version": self._CSV_AUTOFETCH_VERSION,
            })
            self.flush_cache()
            logger.info(
                "Auto-loaded %s CSV prices for %s via %s (%s)",
                len(price_data),
                ticker,
                source_kind,
                source_url,
            )
            return True
        except (URLError, TimeoutError, ValueError) as e:
            logger.warning("Failed auto-load for %s from TASE: %s", ticker, e)
        except Exception as e:
            logger.error("Unexpected auto-load error for %s: %s", ticker, e)

        self._meta_cache_set(marker_key, {
            "status": "failed",
            "date": today_iso,
            "version": self._CSV_AUTOFETCH_VERSION,
        })
        self.flush_cache()
        return False

    def fetch_price_history(self, ticker: str, start: date, end: date) -> pd.Series | None:
        """
        Returns daily closing prices for ticker between start and end.
        Cache key = "price_{ticker}"; stores a single growing time series.
        Only fetches date ranges not already in cache.
        Returns None if ticker is null or all fetches fail.

        For virtual "CSV:<name>" tickers, prices are served from the price cache
        only — yfinance is never called.
        """
        if not ticker:
            return None

        cache_key = f"price_{ticker}"

        # ── CSV virtual tickers: read directly from cache, no yfinance ──────
        if ticker.startswith("CSV:"):
            # Keep TASE-backed CSV tickers fresh (at most once per cooldown window).
            self._maybe_autofetch_csv_prices(ticker)
            stored = self._cache.get(cache_key, {})
            result = {d: v for d, v in stored.items()
                      if start.isoformat() <= d <= end.isoformat()
                      and isinstance(v, (int, float))}
            if not result:
                return None
            s = pd.Series(result)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s.sort_index()

        cached_s, missing = self._series_cache_get(cache_key, start, end)

        for gap_start, gap_end in missing:
            try:
                raw = yf.Ticker(ticker).history(
                    start=gap_start.isoformat(),
                    end=(gap_end + timedelta(days=1)).isoformat())
                if not raw.empty:
                    s = raw["Close"].ffill().bfill()
                    s.index = s.index.tz_localize(None) if s.index.tz else s.index
                    if self._is_agorot_ticker(ticker):
                        s = s / 100.0   # ILA → ILS
                    self._series_cache_set(cache_key, {str(k.date()): v for k, v in s.items()})
            except Exception as e:
                logger.error("Failed to fetch price %s [%s-%s]: %s", ticker, gap_start, gap_end, e)

        if missing:
            self.flush_cache()

        # Re-read from cache
        stored = self._cache.get(cache_key, {})
        result = {d: v for d, v in stored.items()
                  if start.isoformat() <= d <= end.isoformat()
                  and isinstance(v, (int, float))}
        if not result:
            return None
        s = pd.Series(result)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.sort_index()

    def get_price_on_date(self, ticker: str, d: date) -> float | None:
        """Single-date price lookup with ±7 day fallback window. Memoized."""
        if not ticker:
            return None
        memo_key = f"price_{ticker}_{d.isoformat()}"
        if memo_key in self._rate_memo:  # reuse the same scalar memo dict
            v = self._rate_memo[memo_key]
            return None if v == -1.0 else v
        s = self.fetch_price_history(ticker, d - timedelta(days=10), d + timedelta(days=1))
        if s is None or s.empty:
            self._rate_memo[memo_key] = -1.0  # sentinel for None
            return None
        s.index = pd.to_datetime(s.index).tz_localize(None)
        target = pd.Timestamp(d)
        past = s[s.index <= target]
        price = float(past.iloc[-1]) if not past.empty else float(s.iloc[0])
        self._rate_memo[memo_key] = price
        return price

    # ─────────────────────────────────────────────────────────────────────────
    # DIVIDEND HISTORY
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_dividend_history(self, ticker: str, start: date, end: date) -> pd.Series:
        """
        Returns a sparse Series of dividend payments (date → amount per share).
        Cache key = "div_{ticker}"; stores the full known dividend history.
        Dividends are historically stable — only today's data needs refreshing.
        CSV virtual tickers have no yfinance dividends — returns empty Series.
        """
        if not ticker:
            return pd.Series(dtype=float)
        if ticker.startswith("CSV:"):
            return pd.Series(dtype=float)

        cache_key = f"div_{ticker}"
        stored = self._cache.get(cache_key, {})
        has_cached = any(isinstance(v, (int, float)) for v in stored.values())

        # Dividends are sparse; if we already have cache, avoid network on every API call.
        if (not has_cached) or self._is_dividend_refresh_due(ticker):
            try:
                divs = yf.Ticker(ticker).dividends
                if not divs.empty:
                    divs.index = divs.index.tz_localize(None) if divs.index.tz else divs.index
                    if self._is_agorot_ticker(ticker):
                        divs = divs / 100.0   # ILA → ILS
                    self._series_cache_set(cache_key,
                                           {str(k.date()): v for k, v in divs.items()})
                self._meta_cache_set(f"div_refresh_status_{ticker}", {
                    "date": date.today().isoformat(),
                    "status": "ok",
                })
            except Exception as e:
                logger.error("Failed to fetch dividends %s: %s", ticker, e)
                self._meta_cache_set(f"div_refresh_status_{ticker}", {
                    "date": date.today().isoformat(),
                    "status": "failed",
                })
            self.flush_cache()

        # Read filtered range from cache
        stored = self._cache.get(cache_key, {})
        result = {d: v for d, v in stored.items()
                  if start.isoformat() <= d <= end.isoformat()
                  and isinstance(v, (int, float))}
        if not result:
            return pd.Series(dtype=float)
        s = pd.Series(result)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.sort_index()

    # ─────────────────────────────────────────────────────────────────────────
    # HOLDINGS REPLAY
    # ─────────────────────────────────────────────────────────────────────────

    def get_holdings_at_date(self, d: date) -> dict[str, float]:
        """
        Returns {asset_name: quantity} snapshot at date d.
        Starts from initial_holdings and replays all buy/sell/split/deposit actions up to d.
        Split actions multiply the existing quantity by split_ratio.

        NIS (ILS cash) is updated dynamically:
          - ILS deposits   → +amount
          - ILS buy        → -amount (cost)
          - ILS sell       → +amount (proceeds)
          - ILS dividends of ILS assets → +net dividend amount  (handled separately;
            dividends are NOT replayed here as they require price history which would
            create a circular dependency — NIS dividend credit is added in
            get_asset_value_history via the cash asset's quantity adjustment)
        """
        ds = d.isoformat()
        if ds in self._holdings_memo:
            return dict(self._holdings_memo[ds])  # return a shallow copy

        holdings = {name: info["quantity"]
                    for name, info in self.initial_holdings.items()}

        # Deduct monthly fees from the appropriate cash asset
        cur_month = self.start_date.replace(day=1)
        while cur_month <= d:
            cfg = self.get_fee_config_at_date(cur_month)
            fee_cur = cfg["monthly_fee_currency"]
            if fee_cur == "ILS":
                holdings["NIS"] = holdings.get("NIS", 0.0) - cfg["monthly_fee"]
            elif fee_cur == "USD" and "USD" in holdings:
                holdings["USD"] = holdings.get("USD", 0.0) - cfg["monthly_fee"]
            cur_month += relativedelta(months=1)

        for action in self.actions:
            if action["date"] > ds:
                break
            atype    = action["action_type"]
            symbol   = action.get("symbol")
            qty      = action.get("quantity") or 0.0
            currency = action.get("currency") or "ILS"
            amount   = action.get("amount") or (
                (action.get("quantity") or 0) * (action.get("cost_per_unit") or 0))
            adate_obj = date.fromisoformat(action["date"])

            if atype == "buy" and symbol:
                # ── Currency exchange: buying a cash asset with another currency ──
                # e.g. symbol="USD", currency="ILS"  →  NIS -= amount+fee, USD += amount/fx
                # e.g. symbol="NIS", currency="USD"  →  USD -= amount+fee, NIS += amount/fx
                src_cash  = self._cash_name_for_currency(currency)   # cash being spent
                dest_cash = self._cash_name_for_currency(None, symbol)  # cash being received (if symbol is a cash asset)
                is_fx_buy = (dest_cash is not None and src_cash is not None
                             and dest_cash != src_cash and symbol in self.initial_holdings
                             and self.initial_holdings[symbol].get("asset_type") == "cash")

                if is_fx_buy and amount:
                    fee = self._calc_transaction_fee(amount, currency, adate_obj)
                    # Debit source cash
                    holdings[src_cash] = holdings.get(src_cash, 0.0) - amount - fee
                    # Credit dest cash: convert amount to destination currency
                    dest_currency = self.initial_holdings[dest_cash].get("currency", dest_cash)
                    fx = self.get_rate_on_date(currency, dest_currency, adate_obj)
                    holdings[dest_cash] = holdings.get(dest_cash, 0.0) + amount * fx
                else:
                    # Normal asset buy — resolve symbol to canonical holding name
                    action_ticker = action.get("ticker")
                    canonical = symbol
                    if symbol not in holdings:
                        # Check if a holding with matching ticker exists
                        for hname, hinfo in self.initial_holdings.items():
                            if action_ticker and hinfo.get("ticker") == action_ticker:
                                canonical = hname
                                break
                    holdings[canonical] = holdings.get(canonical, 0.0) + qty
                    if currency == "ILS" and amount:
                        fee = self._calc_transaction_fee(amount, currency, adate_obj)
                        holdings["NIS"] = holdings.get("NIS", 0.0) - amount - fee
                    elif currency == "USD" and amount and "USD" in holdings:
                        fee = self._calc_transaction_fee(amount, currency, adate_obj)
                        holdings["USD"] = holdings.get("USD", 0.0) - amount - fee
            elif atype == "sell" and symbol:
                action_ticker = action.get("ticker")
                canonical = symbol
                if symbol not in holdings:
                    for hname, hinfo in self.initial_holdings.items():
                        if action_ticker and hinfo.get("ticker") == action_ticker:
                            canonical = hname
                            break
                holdings[canonical] = holdings.get(canonical, 0.0) - qty
                if currency == "ILS" and amount:
                    fee = self._calc_transaction_fee(amount, currency, adate_obj)
                    holdings["NIS"] = holdings.get("NIS", 0.0) + amount - fee
                elif currency == "USD" and amount and "USD" in holdings:
                    fee = self._calc_transaction_fee(amount, currency, adate_obj)
                    holdings["USD"] = holdings.get("USD", 0.0) + amount - fee
            elif atype == "deposit":
                if currency == "ILS" and amount:
                    holdings["NIS"] = holdings.get("NIS", 0.0) + amount
                elif currency == "USD" and amount and "USD" in holdings:
                    holdings["USD"] = holdings.get("USD", 0.0) + amount
            elif atype == "split" and symbol:
                ratio = action.get("split_ratio", 1.0)
                ticker_s = action.get("ticker")
                # Collect all keys that refer to this asset
                keys_to_split = set()
                for key in list(holdings.keys()):
                    if key == symbol or key == ticker_s:
                        keys_to_split.add(key)
                    # Match initial holding by ticker
                    info = self.initial_holdings.get(key, {})
                    if ticker_s and info.get("ticker") == ticker_s:
                        keys_to_split.add(key)
                for key in keys_to_split:
                    holdings[key] = holdings[key] * ratio
            elif atype == "spinoff" and symbol:
                # Spinoff: receive new shares of spun-off company (e.g. SOLV from MMM)
                holdings[symbol] = holdings.get(symbol, 0.0) + qty

        # ── Credit USD cash with net dividends from USD-denominated stocks ────
        usd_divs = self._get_usd_dividend_events()
        usd_credit = 0.0
        for ev_date, net in usd_divs:
            if ev_date > d:
                break
            usd_credit += net
        holdings["USD"] = holdings.get("USD", 0.0) + usd_credit

        self._holdings_memo[ds] = dict(holdings)  # store immutable snapshot
        return holdings

    def _get_usd_dividend_events(self) -> list[tuple[date, float]]:
        """
        Lazily build a sorted list of (div_date, net_amount) for all USD-denominated
        stocks.  Uses action-based qty tracking (no recursive get_holdings_at_date call).
        Cached after first computation.
        """
        if hasattr(self, "_usd_div_events_cache"):
            return self._usd_div_events_cache

        events: list[tuple[date, float]] = []

        for hname, hinfo in self.initial_holdings.items():
            if hinfo.get("asset_type") in ("cash", "bond") or hinfo.get("currency") != "USD":
                continue
            ticker_h = hinfo.get("ticker")
            if not ticker_h or ticker_h.startswith("CSV:"):
                continue
            base_qty = hinfo["quantity"]

            try:
                div_series = self.fetch_dividend_history(ticker_h, self.start_date, self.end_date)
            except Exception:
                continue
            if div_series.empty:
                continue

            # Build a timeline of qty changes from actions (buys/sells/splits/spinoffs)
            qty_changes: list[tuple[date, str, float]] = []
            for act in self.actions:
                at2 = act["action_type"]
                sym2 = act.get("symbol")
                tick2 = act.get("ticker")
                if at2 == "buy" and self._symbol_matches(hname, sym2 or "", tick2):
                    qty_changes.append((date.fromisoformat(act["date"]), "add", act.get("quantity") or 0))
                elif at2 == "sell" and self._symbol_matches(hname, sym2 or "", tick2):
                    qty_changes.append((date.fromisoformat(act["date"]), "sub", act.get("quantity") or 0))
                elif at2 == "split" and self._symbol_matches(hname, sym2 or "", tick2):
                    qty_changes.append((date.fromisoformat(act["date"]), "mul", act.get("split_ratio", 1.0)))
                elif at2 == "spinoff" and self._symbol_matches(hname, sym2 or "", tick2):
                    qty_changes.append((date.fromisoformat(act["date"]), "add", act.get("quantity") or 0))

            for div_ts, div_per_share in div_series.items():
                div_date = div_ts.date() if hasattr(div_ts, "date") else div_ts
                # Calculate qty held on div_date
                q = base_qty
                for chg_date, chg_type, chg_val in qty_changes:
                    if chg_date > div_date:
                        break
                    if chg_type == "add":
                        q += chg_val
                    elif chg_type == "sub":
                        q -= chg_val
                    elif chg_type == "mul":
                        q *= chg_val
                if q <= 0:
                    continue
                cfg = self.get_fee_config_at_date(div_date)
                gross = q * div_per_share
                net = gross * (1 - cfg["dividend_tax_rate"])
                events.append((div_date, net))

        events.sort(key=lambda x: x[0])
        self._usd_div_events_cache = events
        return events

    def _cash_name_for_currency(self, currency: str | None, symbol: str | None = None) -> str | None:
        """
        Returns the holding name for a cash asset matching the given currency or symbol.
        e.g. "ILS" → "NIS",  "USD" → "USD"
        """
        for name, info in self.initial_holdings.items():
            if info.get("asset_type") != "cash":
                continue
            if symbol and name == symbol:
                return name
            if currency and info.get("currency") == currency:
                return name
        return None

    def _symbol_matches(self, holding_name: str, symbol: str, ticker: str | None) -> bool:
        """Returns True if holding_name corresponds to the same asset as symbol/ticker."""
        if holding_name == symbol:
            return True
        # Check via initial_holdings
        info = self.initial_holdings.get(holding_name, {})
        if ticker and info.get("ticker") == ticker:
            return True
        return False

    def get_splits(self) -> list[dict]:
        """Returns all split actions sorted by date."""
        return [a for a in self.actions if a["action_type"] == "split"]

    def get_cumulative_split_ratio(self, symbol: str, ticker: str | None,
                                   from_date: date, to_date: date) -> float:
        """
        Returns the combined split multiplier for an asset between two dates.
        E.g. two 2-for-1 splits → 4.0; no splits → 1.0.
        Used to adjust historical cost basis per unit.
        """
        ratio = 1.0
        for action in self.actions:
            if action["action_type"] != "split":
                continue
            adate = date.fromisoformat(action["date"])
            if not (from_date <= adate <= to_date):
                continue
            if action.get("symbol") == symbol or action.get("ticker") == ticker:
                ratio *= action.get("split_ratio", 1.0)
        return ratio

    # ─────────────────────────────────────────────────────────────────────────
    # TRANSACTION FEE CALCULATOR
    # ─────────────────────────────────────────────────────────────────────────

    def _calc_transaction_fee(self, amount: float, currency: str, d: date) -> float:
        """Returns the transaction fee in the action's own currency."""
        cfg = self.get_fee_config_at_date(d)
        rate = cfg.get(f"{currency}_rate", cfg["ILS_rate"])
        minimum = cfg.get(f"{currency}_minimum", cfg["ILS_minimum"])
        return max(amount * rate, minimum)

    # ─────────────────────────────────────────────────────────────────────────
    # PER-ASSET VALUE HISTORY  (core method)
    # ─────────────────────────────────────────────────────────────────────────

    def get_asset_value_history(self,
                                name: str,
                                holding_info: dict,
                                display_currency: str,
                                dates: list[date]) -> dict:
        """
        Computes the full value history of ONE asset across the given dates.

        For each date returns:
          price_value       = quantity × price × fx_rate
          gross_dividends   = cumulative dividends received (before tax)
          dividend_tax      = tax paid on dividends
          net_dividends     = gross_dividends - dividend_tax
          total_value       = price_value + net_dividends   ← after-tax

        Also returns:
          dividend_events   = list of individual dividend payments with full detail
          ticker_currency   = native currency of the ticker
          has_price_data    = False if ticker is null
        """
        ticker   = holding_info.get("ticker")
        asset_currency = holding_info.get("currency") or "ILS"
        asset_type = holding_info.get("asset_type", "stock")

        memo_key = (
            "asset_value_history",
            name,
            ticker,
            asset_currency,
            asset_type,
            display_currency,
            tuple(d.isoformat() for d in dates),
        )
        if memo_key in self._result_memo:
            return self._result_memo[memo_key]

        result = {
            "name": name,
            "ticker": ticker,
            "asset_type": asset_type,
            "has_price_data": ticker is not None,
            "ticker_currency": None,
            "timeline": [],
            "dividend_events": [],
        }

        if not dates:
            return result

        start = dates[0]
        end   = dates[-1]

        # ── Determine ticker's native trading currency ──────────────────────
        ticker_currency = asset_currency  # fallback
        if ticker:
            ticker_currency = self.get_ticker_currency(ticker)
        result["ticker_currency"] = ticker_currency

        # ── Fetch price history ─────────────────────────────────────────────
        price_series = None
        if asset_type == "cash":
            # Cash: price = 1 in its own currency
            price_series = pd.Series(1.0,
                                     index=pd.DatetimeIndex(
                                         [pd.Timestamp(d) for d in dates]))
            ticker_currency = asset_currency
        elif ticker:
            price_series = self.fetch_price_history(ticker, start, end)

        # ── Fetch FX rate: ticker_currency → display_currency ───────────────
        fx_series = self.fetch_exchange_rate_history(
            ticker_currency, display_currency, start, end)
        fx_series.index = pd.to_datetime(fx_series.index).tz_localize(None)

        # ── Fetch dividend history ──────────────────────────────────────────
        div_series = self.fetch_dividend_history(ticker, start, end) if ticker else pd.Series(dtype=float)

        # ── Build dividend events with tax applied ──────────────────────────
        cumulative_gross_div = 0.0
        cumulative_tax_div   = 0.0
        # Map div_date → cumulative net by that date
        div_cumulative_by_date: dict[date, tuple[float, float, float]] = {}

        for div_ts, div_per_share in div_series.items():
            div_date = div_ts.date() if hasattr(div_ts, "date") else div_ts
            cfg = self.get_fee_config_at_date(div_date)
            tax_rate = cfg["dividend_tax_rate"]

            # quantity held on dividend date
            holdings_snap = self.get_holdings_at_date(div_date)
            # Map by ticker since holdings use name but buys add by symbol
            qty = self._resolve_quantity(name, ticker, holdings_snap)

            if qty <= 0:
                continue

            # FX rate on dividend date
            fx_rate = self.get_rate_on_date(ticker_currency, display_currency, div_date)

            gross = qty * div_per_share * fx_rate
            tax   = gross * tax_rate
            net   = gross - tax

            cumulative_gross_div += gross
            cumulative_tax_div   += tax

            result["dividend_events"].append({
                "date":            div_date.isoformat(),
                "per_share":       round(div_per_share, 6),
                "quantity_held":   qty,
                "gross_amount":    round(gross, 2),
                "tax":             round(tax, 2),
                "net_amount":      round(net, 2),
                "currency":        display_currency,
                "tax_rate":        tax_rate,
            })
            div_cumulative_by_date[div_date] = (cumulative_gross_div,
                                                 cumulative_tax_div,
                                                 cumulative_gross_div - cumulative_tax_div)

        # Build lookup: for a given date, latest cumulative div values
        sorted_div_dates = sorted(div_cumulative_by_date.keys())
        sorted_div_ordinals = [d.toordinal() for d in sorted_div_dates]

        def _get_cumulative_divs(d: date):
            idx = bisect_right(sorted_div_ordinals, d.toordinal()) - 1
            if idx < 0:
                return 0.0, 0.0, 0.0
            return div_cumulative_by_date[sorted_div_dates[idx]]

        # Pre-align series to requested dates once (avoid repeated index conversions/slices).
        date_index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
        aligned_prices = None
        if price_series is not None:
            ps = price_series.copy()
            ps.index = pd.to_datetime(ps.index).tz_localize(None)
            aligned_prices = ps.reindex(date_index, method="ffill")
        fxs = fx_series.copy()
        fxs.index = pd.to_datetime(fxs.index).tz_localize(None)
        aligned_fx = fxs.reindex(date_index, method="ffill").fillna(1.0)

        # ── Build timeline ──────────────────────────────────────────────────
        is_nis = (name == "NIS" and asset_type == "cash" and asset_currency == "ILS")
        is_usd = (name == "USD" and asset_type == "cash" and asset_currency == "USD")
        for d in dates:
            # quantity on this date
            holdings_snap = self.get_holdings_at_date(d)
            if asset_type == "cash":
                qty = holdings_snap.get(name, 0.0)  # raw, not clamped
            else:
                qty = self._resolve_quantity(name, ticker, holdings_snap)
            # For NIS: add cumulative net ILS dividends received up to this date
            if is_nis:
                qty = qty + self.get_ils_dividends_up_to(d)
            # USD dividends are now credited in get_holdings_at_date

            # price in ticker's native currency
            price = None
            if aligned_prices is not None:
                p = aligned_prices.iloc[len(result["timeline"])]
                price = float(p) if pd.notna(p) else None

            # FX rate on this date
            fx_val = aligned_fx.iloc[len(result["timeline"])]
            fx_rate = float(fx_val) if pd.notna(fx_val) else 1.0

            price_value = round(qty * price * fx_rate, 2) if price is not None else None

            gross_div, tax_div, net_div = _get_cumulative_divs(d)

            total_value = None
            if price_value is not None:
                total_value = round(price_value + net_div, 2)

            result["timeline"].append({
                "date":                    d.isoformat(),
                "quantity":                qty,
                "price":                   round(price, 4) if price else None,
                "price_in_display_currency": round(price * fx_rate, 4) if price else None,
                "fx_rate":                 round(fx_rate, 6),
                "price_value":             price_value,
                "gross_cumulative_dividends": round(gross_div, 2),
                "dividend_tax":            round(tax_div, 2),
                "net_cumulative_dividends":  round(net_div, 2),
                "total_value":             total_value,
                "display_currency":        display_currency,
            })

        self._result_memo[memo_key] = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # RESOLVE QUANTITY HELPER
    # ─────────────────────────────────────────────────────────────────────────

    def _build_dividends_series(self, target_currency: str) -> list[tuple[date, float]]:
        """
        Build a sorted list of (div_date, net_amount) for all assets whose
        currency matches target_currency.  Called once per currency and cached.
        """
        events: list[tuple[date, float]] = []
        assets_to_check: dict[str, str] = {}  # name → ticker

        for name, info in self.initial_holdings.items():
            if info.get("currency") != target_currency:
                continue
            ticker = info.get("ticker")
            if not ticker or ticker == "USDILS=X":
                continue
            assets_to_check[name] = ticker

        # For USD: also pick up action-only USD assets
        if target_currency == "USD":
            for action in self.actions:
                if action["action_type"] == "buy" and action.get("currency") == "USD":
                    sym = action.get("symbol")
                    tkr = action.get("ticker")
                    if sym and tkr and sym not in self.initial_holdings and sym not in assets_to_check:
                        assets_to_check[sym] = tkr

        for name, ticker in assets_to_check.items():
            ticker_currency = self.get_ticker_currency(ticker)
            div_series = self.fetch_dividend_history(ticker, self.start_date, self.end_date)
            for div_ts, div_per_share in div_series.items():
                div_date = div_ts.date() if hasattr(div_ts, "date") else div_ts
                cfg = self.get_fee_config_at_date(div_date)
                holdings_snap = self.get_holdings_at_date(div_date)
                qty = self._resolve_quantity(name, ticker, holdings_snap)
                if qty <= 0:
                    continue
                fx = self.get_rate_on_date(ticker_currency, target_currency, div_date)
                gross = qty * div_per_share * fx
                net = gross * (1 - cfg["dividend_tax_rate"])
                events.append((div_date, net))

        events.sort(key=lambda x: x[0])
        return events

    def _cumulative_dividends_up_to(self, target_currency: str, d: date) -> float:
        """
        Returns cumulative net dividends in target_currency up to date d.
        Builds the full event list once (cached per currency) then binary-searches.
        """
        cache_attr = f"_div_events_{target_currency}"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, self._build_dividends_series(target_currency))
        events: list[tuple[date, float]] = getattr(self, cache_attr)
        total = 0.0
        for ev_date, net in events:
            if ev_date <= d:
                total += net
            else:
                break  # list is sorted
        return total

    def get_ils_dividends_up_to(self, d: date) -> float:
        return self._cumulative_dividends_up_to("ILS", d)

    def get_usd_dividends_up_to(self, d: date) -> float:
        return self._cumulative_dividends_up_to("USD", d)

    def _resolve_quantity(self, name: str, ticker: str | None,
                          holdings_snap: dict) -> float:
        """
        holdings_snap keys can be asset names (initial) or symbols (from actions).
        Try name first, then ticker symbol.
        """
        if name in holdings_snap:
            return max(holdings_snap[name], 0.0)
        if ticker and ticker in holdings_snap:
            return max(holdings_snap[ticker], 0.0)
        # Try matching by symbol across actions
        for action in self.actions:
            if action.get("ticker") == ticker and action.get("symbol") in holdings_snap:
                return max(holdings_snap[action["symbol"]], 0.0)
        return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # PORTFOLIO TIMELINE  (sum of all assets)
    # ─────────────────────────────────────────────────────────────────────────

    def get_portfolio_timeline(self,
                               display_currency: str = "ILS",
                               start: date | None = None,
                               end: date | None = None,
                               frequency: str = "W") -> dict:
        """
        Builds the full portfolio value timeline.

        frequency: 'D'=daily, 'W'=weekly (default), 'M'=monthly

        Returns:
          {
            "timeline": [ { date, total_value, net_dividends, deposits, assets:{...} } ],
            "display_currency": "ILS",
            "assets_without_prices": [ list of names ]
          }
        """
        start = start or self.start_date
        end   = end   or self.end_date

        memo_key = ("portfolio_timeline", display_currency, start.isoformat(), end.isoformat(), frequency)
        if memo_key in self._result_memo:
            return self._result_memo[memo_key]

        # Generate date range
        dates = self._date_range(start, end, frequency)

        # Helper: find the first date an asset appears in the portfolio
        def _asset_start_dates(name: str, info: dict) -> list[date]:
            """Return dates filtered to start from when the asset first exists."""
            if info.get("quantity", 0) > 0:
                return dates  # initial holding with shares — exists from start
            # Find earliest buy or spinoff action for this asset
            ticker = info.get("ticker")
            for action in self.actions:
                if action["action_type"] not in ("buy", "spinoff"):
                    continue
                if action.get("symbol") == name or (ticker and action.get("ticker") == ticker):
                    first = date.fromisoformat(action["date"])
                    return [d for d in dates if d >= first]
            return dates  # fallback

        # Fetch all asset histories
        all_asset_histories = {}
        for name, info in self.initial_holdings.items():
            logger.info("Fetching history for %s (%s)...", name, info.get("ticker"))
            asset_dates = _asset_start_dates(name, info)
            all_asset_histories[name] = self.get_asset_value_history(
                name, info, display_currency, asset_dates)

        # Also fetch histories for assets acquired via buy actions
        # (skip if same ticker already covered by initial_holdings)
        existing_tickers_tl = {info.get("ticker") for info in self.initial_holdings.values() if info.get("ticker")}
        action_assets: dict[str, dict] = {}
        for action in self.actions:
            if action["action_type"] == "buy" and action.get("ticker"):
                sym = action["symbol"]
                tkr = action["ticker"]
                if sym not in all_asset_histories and sym not in action_assets:
                    if tkr in existing_tickers_tl:
                        continue  # already tracked under initial_holdings name
                    action_assets[sym] = {
                        "quantity": 0.0,
                        "ticker": tkr,
                        "currency": action["currency"],
                        "asset_type": "stock",
                    }
        for name, info in action_assets.items():
            logger.info("Fetching history for action asset %s (%s)...", name, info["ticker"])
            asset_dates = _asset_start_dates(name, info)
            all_asset_histories[name] = self.get_asset_value_history(
                name, info, display_currency, asset_dates)

        # Build cumulative deposits per date
        deposit_by_date = self._get_deposits_by_date(display_currency, dates)

        # Assemble timeline
        # Build date→index maps for each asset (assets may have different date ranges)
        asset_date_maps: dict[str, dict[str, int]] = {}
        for aname, hist in all_asset_histories.items():
            asset_date_maps[aname] = {
                p["date"]: idx for idx, p in enumerate(hist["timeline"])
            }

        timeline = []
        for i, d in enumerate(dates):
            total_value        = 0.0
            total_net_divs     = 0.0
            assets_snapshot    = {}
            has_partial        = False
            d_iso = d.isoformat()

            for aname, hist in all_asset_histories.items():
                idx = asset_date_maps[aname].get(d_iso)
                if idx is None or not hist["timeline"]:
                    continue
                point = hist["timeline"][idx]
                tv = point["total_value"]
                if tv is None:
                    has_partial = True
                else:
                    total_value    += tv
                    total_net_divs += point["net_cumulative_dividends"]
                assets_snapshot[aname] = {
                    "quantity":     point["quantity"],
                    "price_value":  point["price_value"],
                    "net_dividends": point["net_cumulative_dividends"],
                    "total_value":  point["total_value"],
                }

            timeline.append({
                "date":             d.isoformat(),
                "total_value":      round(total_value, 2),
                "net_dividends":    round(total_net_divs, 2),
                "cumulative_deposits": deposit_by_date.get(d, 0.0),
                "has_missing_prices": has_partial,
                "display_currency": display_currency,
                "assets":           assets_snapshot,
            })

        assets_without_prices = [
            n for n, h in all_asset_histories.items() if not h["has_price_data"]
        ]

        result = {
            "timeline":              timeline,
            "display_currency":      display_currency,
            "assets_without_prices": assets_without_prices,
        }
        self._result_memo[memo_key] = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # ASSET GAIN SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def get_asset_gain_summary(self,
                               name: str,
                               holding_info: dict,
                               display_currency: str) -> dict:
        """
        Per-asset gain breakdown:
          cost_basis          = total amount paid to buy this asset
          current_value       = quantity × current price × fx rate
          unrealized_gain     = current_value - cost_basis
          net_dividends       = cumulative dividends after tax
          transaction_fees    = fees paid on buy/sell actions for this asset
          total_net_gain      = unrealized_gain + net_dividends - transaction_fees
          gain_pct            = total_net_gain / cost_basis × 100
        """
        ticker = holding_info.get("ticker")
        memo_key = ("asset_gain_summary", name, ticker, display_currency)
        if memo_key in self._result_memo:
            return self._result_memo[memo_key]
        today  = self.end_date

        # ── Cost basis ──────────────────────────────────────────────────────
        cost_basis = 0.0
        cost_basis_per_unit = holding_info.get("cost_basis_per_unit")
        initial_qty = holding_info.get("quantity", 0.0)

        if cost_basis_per_unit and initial_qty:
            # Adjust cost_per_unit for any splits between start_date and today
            split_ratio = self.get_cumulative_split_ratio(name, ticker, self.start_date, today)
            adjusted_cpu = cost_basis_per_unit / split_ratio if split_ratio else cost_basis_per_unit
            rate = self.get_rate_on_date(holding_info.get("currency") or "ILS",
                                         display_currency, self.start_date)
            cost_basis += initial_qty * adjusted_cpu * rate

        # Add buy actions — adjust cost_per_unit for splits that happened after the buy
        transaction_fees = 0.0

        for action in self.actions:
            if action.get("symbol") != name and action.get("ticker") != ticker:
                continue
            atype = action["action_type"]
            if atype not in ("buy", "sell"):
                continue
            adate  = date.fromisoformat(action["date"])
            rate   = self.get_rate_on_date(action.get("currency") or "ILS",
                                           display_currency, adate)
            amount = action.get("amount") or (
                (action.get("quantity") or 0) * (action.get("cost_per_unit") or 0))

            if atype == "buy":
                # For cost basis: the total amount paid is not affected by later splits
                cost_basis += amount * rate
                fee = self._calc_transaction_fee(amount, action.get("currency") or "ILS", adate)
                transaction_fees += fee * rate

            elif atype == "sell":
                fee = self._calc_transaction_fee(amount, action.get("currency") or "ILS", adate)
                transaction_fees += fee * rate

        # ── Current value ───────────────────────────────────────────────────
        today_holdings = self.get_holdings_at_date(today)
        if holding_info.get("asset_type") == "cash":
            current_qty = today_holdings.get(name, 0.0)  # raw, not clamped
        else:
            current_qty = self._resolve_quantity(name, ticker, today_holdings)
        # NIS cash: add cumulative ILS dividends
        if name == "NIS" and holding_info.get("asset_type") == "cash" and holding_info.get("currency") == "ILS":
            current_qty = current_qty + self.get_ils_dividends_up_to(today)
        # USD dividends are now credited in get_holdings_at_date
        current_price = self.get_price_on_date(ticker, today) if ticker else None
        current_fx    = self.get_rate_on_date(
            self.get_ticker_currency(ticker) if ticker else holding_info.get("currency") or "ILS",
            display_currency, today)

        current_value = round(current_qty * current_price * current_fx, 2) \
            if current_price is not None else None

        unrealized_gain = round(current_value - cost_basis, 2) \
            if current_value is not None else None

        # ── Dividends (after tax) ───────────────────────────────────────────
        ticker_currency = self.get_ticker_currency(ticker) if ticker else holding_info.get("currency") or "ILS"
        div_series = self.fetch_dividend_history(ticker, self.start_date, today) if ticker else pd.Series(dtype=float)

        net_dividends = 0.0
        for div_ts, div_per_share in div_series.items():
            div_date = div_ts.date() if hasattr(div_ts, "date") else div_ts
            cfg      = self.get_fee_config_at_date(div_date)
            qty_snap = self._resolve_quantity(name, ticker, self.get_holdings_at_date(div_date))
            if qty_snap <= 0:
                continue
            fx   = self.get_rate_on_date(ticker_currency, display_currency, div_date)
            gross = qty_snap * div_per_share * fx
            net_dividends += gross * (1 - cfg["dividend_tax_rate"])

        # ── Capital gains tax on realised sells ─────────────────────────────
        capital_gains_tax = 0.0
        running_cost_per_unit = (cost_basis_per_unit or 0.0)  # simplified avg
        for action in self.actions:
            if action.get("symbol") != name and action.get("ticker") != ticker:
                continue
            if action["action_type"] != "sell":
                continue
            adate = date.fromisoformat(action["date"])
            cfg   = self.get_fee_config_at_date(adate)
            rate  = self.get_rate_on_date(action.get("currency") or "ILS", display_currency, adate)
            sell_amount = (action.get("amount") or 0) * rate
            qty_sold    = action.get("quantity") or 0
            cost_of_sold = qty_sold * running_cost_per_unit
            gain = sell_amount - cost_of_sold
            if gain > 0:
                capital_gains_tax += gain * cfg["capital_gains_tax_rate"]

        total_net_gain = None
        gain_pct       = None
        if unrealized_gain is not None:
            total_net_gain = round(unrealized_gain + net_dividends
                                   - transaction_fees - capital_gains_tax, 2)
            if cost_basis > 0:
                gain_pct = round(total_net_gain / cost_basis * 100, 2)

        result = {
            "asset":                name,
            "ticker":               ticker,
            "current_quantity":     current_qty,
            "cost_basis":           round(cost_basis, 2),
            "current_value":        current_value,
            "unrealized_gain":      unrealized_gain,
            "net_dividends":        round(net_dividends, 2),
            "transaction_fees":     round(transaction_fees, 2),
            "capital_gains_tax":    round(capital_gains_tax, 2),
            "total_net_gain":       total_net_gain,
            "gain_pct":             gain_pct,
            "display_currency":     display_currency,
        }
        self._result_memo[memo_key] = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # PORTFOLIO-LEVEL FEES & TAXES
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_fees_and_taxes(self, display_currency: str = "ILS") -> dict:
        """
        Computes all portfolio-level costs. Dividend tax is NOT recalculated
        here — it is summed from asset-level data.

        Returns:
          total_monthly_fees, total_transaction_fees,
          total_capital_gains_tax, total_dividend_tax (from assets),
          total_costs, breakdown[]
        """
        memo_key = ("fees_and_taxes", display_currency)
        if memo_key in self._result_memo:
            return self._result_memo[memo_key]

        breakdown = []
        total_monthly_fees     = 0.0
        total_transaction_fees = 0.0
        total_capital_gains_tax = 0.0

        # ── Monthly fees ────────────────────────────────────────────────────
        cur = self.start_date.replace(day=1)
        today_month = self.end_date.replace(day=1)
        while cur <= today_month:
            cfg   = self.get_fee_config_at_date(cur)
            fee   = cfg["monthly_fee"]
            fee_currency = cfg["monthly_fee_currency"]
            rate  = self.get_rate_on_date(fee_currency, display_currency, cur)
            fee_display = fee * rate
            total_monthly_fees += fee_display
            breakdown.append({
                "date":     cur.isoformat(),
                "type":     "monthly_fee",
                "amount":   round(fee_display, 2),
                "currency": display_currency,
            })
            cur += relativedelta(months=1)

        # ── Transaction fees & capital-gains tax ────────────────────────────
        # Track average cost basis per asset for CG calc
        avg_cost: dict[str, float] = {}    # symbol → avg cost per unit in native currency

        for action in self.actions:
            atype  = action["action_type"]
            symbol = action.get("symbol")
            adate  = date.fromisoformat(action["date"])
            currency = action.get("currency") or "ILS"
            rate   = self.get_rate_on_date(currency, display_currency, adate)
            amount = action.get("amount") or (
                (action.get("quantity") or 0) * (action.get("cost_per_unit") or 0))
            qty    = action.get("quantity") or 0

            if atype in ("buy", "sell") and amount:
                fee = self._calc_transaction_fee(amount, currency, adate)
                fee_display = fee * rate
                total_transaction_fees += fee_display
                breakdown.append({
                    "date":     adate.isoformat(),
                    "type":     "transaction_fee",
                    "symbol":   symbol,
                    "amount":   round(fee_display, 2),
                    "currency": display_currency,
                })

            if atype == "buy" and qty:
                # Update running average cost
                prev_qty  = self._resolve_quantity(
                    symbol or "", action.get("ticker"), self.get_holdings_at_date(adate)) - qty
                prev_qty  = max(prev_qty, 0)
                prev_cost = avg_cost.get(symbol, 0.0)
                cost_per_unit = amount / qty if qty else 0
                new_qty   = prev_qty + qty
                avg_cost[symbol] = ((prev_qty * prev_cost + qty * cost_per_unit)
                                    / new_qty) if new_qty else cost_per_unit

            if atype == "sell" and qty and symbol:
                cfg   = self.get_fee_config_at_date(adate)
                sell_price_per_unit = amount / qty if qty else 0
                cost_per_unit = avg_cost.get(symbol, 0.0)
                gain_per_unit = sell_price_per_unit - cost_per_unit
                if gain_per_unit > 0:
                    cg_tax = qty * gain_per_unit * cfg["capital_gains_tax_rate"] * rate
                    total_capital_gains_tax += cg_tax
                    breakdown.append({
                        "date":     adate.isoformat(),
                        "type":     "capital_gains_tax",
                        "symbol":   symbol,
                        "amount":   round(cg_tax, 2),
                        "currency": display_currency,
                    })

        # ── Dividend tax: sum from asset histories ───────────────────────────
        total_dividend_tax = 0.0
        for name, info in self.initial_holdings.items():
            ticker = info.get("ticker")
            if not ticker:
                continue
            ticker_currency = self.get_ticker_currency(ticker)
            div_series = self.fetch_dividend_history(ticker, self.start_date, self.end_date)
            for div_ts, div_per_share in div_series.items():
                div_date = div_ts.date() if hasattr(div_ts, "date") else div_ts
                cfg  = self.get_fee_config_at_date(div_date)
                qty  = self._resolve_quantity(name, ticker, self.get_holdings_at_date(div_date))
                if qty <= 0:
                    continue
                fx   = self.get_rate_on_date(ticker_currency, display_currency, div_date)
                gross = qty * div_per_share * fx
                tax   = gross * cfg["dividend_tax_rate"]
                total_dividend_tax += tax
                breakdown.append({
                    "date":     div_date.isoformat(),
                    "type":     "dividend_tax",
                    "symbol":   name,
                    "amount":   round(tax, 2),
                    "currency": display_currency,
                })

        breakdown.sort(key=lambda x: x["date"])

        total_costs = (total_monthly_fees + total_transaction_fees
                       + total_capital_gains_tax + total_dividend_tax)

        result = {
            "total_monthly_fees":      round(total_monthly_fees, 2),
            "total_transaction_fees":  round(total_transaction_fees, 2),
            "total_capital_gains_tax": round(total_capital_gains_tax, 2),
            "total_dividend_tax":      round(total_dividend_tax, 2),
            "total_costs":             round(total_costs, 2),
            "display_currency":        display_currency,
            "breakdown":               breakdown,
        }
        self._result_memo[memo_key] = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # CURRENT HOLDINGS SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def get_current_holdings_summary(self, display_currency: str = "ILS") -> dict:
        """
        Returns current portfolio snapshot with latest prices.
        """
        memo_key = ("current_holdings_summary", display_currency)
        if memo_key in self._result_memo:
            return self._result_memo[memo_key]

        today    = self.end_date
        holdings = self.get_holdings_at_date(today)
        total    = 0.0
        assets   = []

        all_infos = dict(self.initial_holdings)
        # Build a set of tickers already covered by initial_holdings
        existing_tickers = {info.get("ticker") for info in all_infos.values() if info.get("ticker")}
        # Add action-only assets (skip if same ticker already exists in initial_holdings)
        for action in self.actions:
            if action["action_type"] == "buy" and action.get("symbol") not in all_infos:
                tkr = action.get("ticker")
                if tkr and tkr in existing_tickers:
                    continue  # already tracked under a different name in initial_holdings
                all_infos[action["symbol"]] = {
                    "quantity": 0.0,
                    "ticker": tkr,
                    "currency": action.get("currency") or "ILS",
                    "asset_type": "stock",
                }

        for name, info in all_infos.items():
            asset_type_s = info.get("asset_type", "stock")
            # For cash assets use raw (possibly negative) balance; others clamp to 0
            if asset_type_s == "cash":
                qty = holdings.get(name, info.get("quantity", 0.0))
            else:
                qty = self._resolve_quantity(name, info.get("ticker"), holdings)
            # NIS cash: also include cumulative ILS dividends received
            if name == "NIS" and asset_type_s == "cash" and info.get("currency") == "ILS":
                qty = qty + self.get_ils_dividends_up_to(today)
            # USD cash: also include cumulative USD dividends received
            # USD dividends are now credited in get_holdings_at_date
            if qty <= 0:
                continue
            ticker = info.get("ticker")
            asset_type = info.get("asset_type", "stock")
            holding_currency = info.get("currency") or "ILS"

            if asset_type == "cash":
                # Cash: price = 1 in the holding's own currency (e.g. USD for USD cash)
                # Use holding_currency (not ticker_currency) so USD cash × USD→ILS rate
                fx = self.get_rate_on_date(holding_currency, display_currency, today)
                price = 1.0
                value = round(qty * fx, 2)
                ticker_currency = holding_currency
            elif ticker:
                ticker_currency = self.get_ticker_currency(ticker)
                fx = self.get_rate_on_date(ticker_currency, display_currency, today)
                price = self.get_price_on_date(ticker, today)
                value = round(qty * price * fx, 2) if price is not None else None
            else:
                ticker_currency = holding_currency
                price = None
                value = None

            if value is not None:
                total += value
            assets.append({
                "name":             name,
                "ticker":           ticker,
                "quantity":         qty,
                "price":            round(price, 4) if price is not None else None,
                "value":            value,
                "currency":         ticker_currency,
                "display_currency": display_currency,
                "asset_type":       asset_type,
                "has_price_data":   ticker is not None or asset_type == "cash",
            })

        assets.sort(key=lambda x: (x["value"] or 0), reverse=True)
        result = {
            "date":             today.isoformat(),
            "total_value":      round(total, 2),
            "display_currency": display_currency,
            "assets":           assets,
        }
        self._result_memo[memo_key] = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # DEPOSITS TIMELINE
    # ─────────────────────────────────────────────────────────────────────────

    def _get_deposits_by_date(self, display_currency: str,
                              dates: list[date]) -> dict[date, float]:
        """Returns cumulative deposits in display_currency at each date."""
        deposit_events: list[tuple[date, float]] = []
        for action in self.actions:
            if action["action_type"] == "deposit" and action.get("amount"):
                d    = date.fromisoformat(action["date"])
                rate = self.get_rate_on_date(action.get("currency") or "ILS",
                                             display_currency, d)
                deposit_events.append((d, action["amount"] * rate))

        deposit_events.sort()
        result: dict[date, float] = {}
        cumulative = 0.0
        ei = 0
        for d in dates:
            while ei < len(deposit_events) and deposit_events[ei][0] <= d:
                cumulative += deposit_events[ei][1]
                ei += 1
            result[d] = round(cumulative, 2)
        return result

    def get_deposits_timeline(self, display_currency: str = "ILS",
                              frequency: str = "W") -> list[dict]:
        """Public method returning deposit timeline."""
        dates  = self._date_range(self.start_date, self.end_date, frequency)
        by_date = self._get_deposits_by_date(display_currency, dates)
        return [{"date": d.isoformat(), "cumulative_deposits": v}
                for d, v in by_date.items()]

    # ─────────────────────────────────────────────────────────────────────────
    # PORTFOLIO RETURN / YEARLY YIELD
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _market_value_from_portfolio_point(point: dict) -> tuple[float, bool]:
        """
        Returns the market value represented by a portfolio timeline point.

        Uses asset.price_value (not asset.total_value) so dividends are not
        double-counted: paid dividends already live in the cash balances.
        """
        total = 0.0
        has_missing = False
        for asset in point.get("assets", {}).values():
            price_value = asset.get("price_value")
            if price_value is None:
                if asset.get("total_value") is None:
                    has_missing = True
                continue
            total += price_value
        return round(total, 2), has_missing

    @staticmethod
    def _cumulative_net_dividends_from_portfolio_point(point: dict) -> float:
        total = 0.0
        for asset in point.get("assets", {}).values():
            total += asset.get("net_dividends", 0.0) or 0.0
        return round(total, 2)

    def _capital_gains_tax_by_date(self,
                                   display_currency: str,
                                   start: date,
                                   end: date) -> dict[date, float]:
        """
        Returns realised capital-gains tax by date in display_currency.

        Monthly fees, transaction fees and dividend tax are already reflected in
        holdings / cash balances. Capital-gains tax is currently calculated as a
        liability but not deducted from holdings, so we subtract it explicitly
        for net return calculations.
        """
        fees = self.calculate_fees_and_taxes(display_currency)
        result: dict[date, float] = {}
        for item in fees.get("breakdown", []):
            if item.get("type") != "capital_gains_tax":
                continue
            d = date.fromisoformat(item["date"])
            if start <= d <= end:
                result[d] = result.get(d, 0.0) + float(item.get("amount") or 0.0)
        return result

    def _external_flows_by_date(self,
                                display_currency: str,
                                start: date,
                                end: date) -> dict[date, dict]:
        """
        External flows relevant to return measurement.

        Included:
          - deposits          (positive for TWR, negative for XIRR)
          - withdrawals       (supported if such actions are added later)
          - realised capital-gains tax (negative external flow for net return)

        Excluded:
          - buys/sells/splits          (internal reallocations)
          - monthly / transaction fees (already in holdings cash balance)
          - dividend tax               (already reflected by net dividends only)
        """
        flows: dict[date, dict] = {}

        def ensure(day: date) -> dict:
            if day not in flows:
                flows[day] = {
                    "deposits": 0.0,
                    "withdrawals": 0.0,
                    "capital_gains_tax": 0.0,
                    "net_external_flow": 0.0,
                }
            return flows[day]

        for action in self.actions:
            adate = date.fromisoformat(action["date"])
            if adate < start or adate > end:
                continue
            atype = action.get("action_type")
            amount = action.get("amount") or (
                (action.get("quantity") or 0) * (action.get("cost_per_unit") or 0)
            )
            if not amount:
                continue
            rate = self.get_rate_on_date(action.get("currency") or "ILS", display_currency, adate)
            amount_display = float(amount) * rate

            if atype == "deposit":
                item = ensure(adate)
                item["deposits"] += amount_display
                item["net_external_flow"] += amount_display
            elif atype == "withdrawal":
                item = ensure(adate)
                item["withdrawals"] += amount_display
                item["net_external_flow"] -= amount_display

        for adate, tax in self._capital_gains_tax_by_date(display_currency, start, end).items():
            item = ensure(adate)
            item["capital_gains_tax"] += tax
            item["net_external_flow"] -= tax

        for item in flows.values():
            for k, v in list(item.items()):
                item[k] = round(v, 2)
        return flows

    @staticmethod
    def _xnpv(rate: float, cashflows: list[tuple[date, float]]) -> float:
        if rate <= -1.0:
            return math.inf
        t0 = cashflows[0][0]
        total = 0.0
        for d, amount in cashflows:
            years = (d - t0).days / 365.0
            total += amount / ((1.0 + rate) ** years)
        return total

    @classmethod
    def _solve_xirr(cls, cashflows: list[tuple[date, float]]) -> float | None:
        """Robust bisection-style XIRR solver without extra dependencies."""
        if len(cashflows) < 2:
            return None
        amounts = [amt for _, amt in cashflows]
        if not any(a < 0 for a in amounts) or not any(a > 0 for a in amounts):
            return None

        low = -0.9999
        high = 1.0
        f_low = cls._xnpv(low, cashflows)
        f_high = cls._xnpv(high, cashflows)

        expand_count = 0
        while math.isfinite(f_low) and math.isfinite(f_high) and f_low * f_high > 0 and expand_count < 50:
            high *= 2.0
            f_high = cls._xnpv(high, cashflows)
            expand_count += 1

        if not (math.isfinite(f_low) and math.isfinite(f_high)) or f_low * f_high > 0:
            return None

        for _ in range(200):
            mid = (low + high) / 2.0
            f_mid = cls._xnpv(mid, cashflows)
            if not math.isfinite(f_mid):
                return None
            if abs(f_mid) < 1e-8:
                return mid
            if f_low * f_mid <= 0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        return (low + high) / 2.0

    def _build_return_state(self,
                            start: date,
                            end: date,
                            display_currency: str) -> dict:
        """
        Build reusable daily return state once for [start-1, end].

        This enables fast rolling-window returns via cumulative log sums instead
        of recomputing each point from scratch.
        """
        # Go back 8 extra days so prices/FX have prior trading days to ffill from
        # when the year boundary (Dec 31) falls on a weekend or holiday.
        fetch_start = start - timedelta(days=8)
        valuation_start = start - timedelta(days=1)
        memo_key = ("return_state", valuation_start.isoformat(), end.isoformat(), display_currency)
        if memo_key in self._result_memo:
            return self._result_memo[memo_key]

        dates = self._date_range(valuation_start, end, "D")
        timeline_payload = self.get_portfolio_timeline(
            display_currency=display_currency,
            start=fetch_start,   # fetch extra history for ffill …
            end=end,
            frequency="D",
        )
        timeline = timeline_payload.get("timeline", [])
        # Only keep points that fall within [valuation_start, end] for the state arrays.
        points_by_date = {
            date.fromisoformat(p["date"]): p for p in timeline
            if date.fromisoformat(p["date"]) >= valuation_start
        }

        flows_by_date = self._external_flows_by_date(display_currency, start, end)

        market_values: list[float | None] = []
        cumulative_dividends: list[float] = []
        rt: list[float | None] = [None]
        cumulative_log: list[float] = [0.0]
        invalid_prefix: list[int] = [0]
        has_missing_prices = False
        missing_dates: list[str] = []
        missing_assets: dict[str, dict[str, str | int]] = {}

        for i, d in enumerate(dates):
            point = points_by_date.get(d)
            if point is None:
                market_values.append(None)
                cumulative_dividends.append(0.0)
                if i > 0:
                    rt.append(None)
                    cumulative_log.append(cumulative_log[-1])
                    invalid_prefix.append(invalid_prefix[-1] + 1)
                continue

            mv, missing = self._market_value_from_portfolio_point(point)
            market_values.append(mv)
            cumulative_dividends.append(self._cumulative_net_dividends_from_portfolio_point(point))
            has_missing_prices = has_missing_prices or missing or point.get("has_missing_prices", False)

            # Track exact missing data per asset/date for user-facing explanation.
            date_iso = d.isoformat()
            missing_any_this_day = False
            for aname, asset in point.get("assets", {}).items():
                if asset.get("total_value") is None:
                    missing_any_this_day = True
                    entry = missing_assets.setdefault(aname, {
                        "missing_days": 0,
                        "first_missing": date_iso,
                        "last_missing": date_iso,
                    })
                    entry["missing_days"] = int(entry["missing_days"]) + 1
                    entry["last_missing"] = date_iso
            if missing_any_this_day:
                missing_dates.append(date_iso)

            if i == 0:
                continue

            prev_mv = market_values[i - 1]
            if prev_mv is None or prev_mv <= 0:
                rt.append(None)
                cumulative_log.append(cumulative_log[-1])
                invalid_prefix.append(invalid_prefix[-1] + 1)
                continue

            flow = flows_by_date.get(d, {}).get("net_external_flow", 0.0)
            r = (mv - prev_mv - flow) / prev_mv
            if r <= -0.999999999:
                rt.append(None)
                cumulative_log.append(cumulative_log[-1])
                invalid_prefix.append(invalid_prefix[-1] + 1)
                continue

            rt.append(r)
            cumulative_log.append(cumulative_log[-1] + math.log1p(r))
            invalid_prefix.append(invalid_prefix[-1])

        result = {
            "dates": dates,
            "date_to_index": {d: i for i, d in enumerate(dates)},
            "market_values": market_values,
            "cumulative_dividends": cumulative_dividends,
            "flows_by_date": flows_by_date,
            "rt": rt,
            "cum_log": cumulative_log,
            "invalid_prefix": invalid_prefix,
            "valuation_start": valuation_start,
            "has_missing_prices": has_missing_prices,
            "missing_data": {
                "missing_days": len(missing_dates),
                "first_missing_date": missing_dates[0] if missing_dates else None,
                "last_missing_date": missing_dates[-1] if missing_dates else None,
                "missing_assets": [
                    {
                        "asset": name,
                        "missing_days": int(info["missing_days"]),
                        "first_missing": str(info["first_missing"]),
                        "last_missing": str(info["last_missing"]),
                    }
                    for name, info in sorted(
                        missing_assets.items(),
                        key=lambda kv: (-int(kv[1]["missing_days"]), kv[0]),
                    )
                ],
            },
        }
        self._result_memo[memo_key] = result
        return result

    @staticmethod
    def _window_twr_from_state(state: dict, start_idx: int, end_idx: int) -> float | None:
        """
        Compute compounded return over (start_idx, end_idx] using cumulative logs.
        """
        if end_idx <= start_idx:
            return None
        invalids = state["invalid_prefix"][end_idx] - state["invalid_prefix"][start_idx]
        if invalids > 0:
            return None
        delta_log = state["cum_log"][end_idx] - state["cum_log"][start_idx]
        return math.expm1(delta_log)

    def get_portfolio_return_summary(self,
                                     start: date,
                                     end: date,
                                     display_currency: str = "ILS") -> dict:
        """
        Net portfolio return summary for a period.

        Metrics returned:
          - time-weighted return (TWR)
          - money-weighted return / XIRR
          - net dividend yield

        Uses daily market values and explicit external cash flows.
        """
        if end < start:
            raise ValueError("end date must be on or after start date")

        memo_key = ("portfolio_return_summary", start.isoformat(), end.isoformat(), display_currency)
        if memo_key in self._result_memo:
            return self._result_memo[memo_key]

        state = self._build_return_state(start, end, display_currency)
        dates = state["dates"]
        date_to_index = state["date_to_index"]
        valuation_start = state["valuation_start"]

        if not dates:
            result = {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "display_currency": display_currency,
                "has_data": False,
            }
            self._result_memo[memo_key] = result
            return result

        start_idx = date_to_index.get(valuation_start)
        end_idx = date_to_index.get(end)
        if start_idx is None or end_idx is None:
            result = {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "display_currency": display_currency,
                "has_data": False,
                "has_missing_prices": state["has_missing_prices"],
            }
            self._result_memo[memo_key] = result
            return result

        # Use the last valid (non-None) market value at or before valuation_start
        # as the start_market_value.  Dec 31 is often a holiday/weekend so we
        # look back up to 7 days to find a traded price.
        start_market_value = None
        for si in range(start_idx, max(start_idx - 8, -1), -1):
            if si >= 0 and state["market_values"][si] is not None:
                start_market_value = state["market_values"][si]
                break

        end_market_value = state["market_values"][end_idx]
        # Similarly look back for the end value if Dec 31 falls on a holiday.
        if end_market_value is None:
            for ei in range(end_idx, max(end_idx - 8, -1), -1):
                if ei >= 0 and state["market_values"][ei] is not None:
                    end_market_value = state["market_values"][ei]
                    break

        if start_market_value is None or end_market_value is None:
            result = {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "display_currency": display_currency,
                "has_data": False,
                "has_missing_prices": state["has_missing_prices"],
            }
            self._result_memo[memo_key] = result
            return result

        flows_by_date = state["flows_by_date"]

        twr = self._window_twr_from_state(state, start_idx, end_idx)
        periods_used = max(0, end_idx - start_idx)

        total_deposits = round(sum(v.get("deposits", 0.0) for v in flows_by_date.values()), 2)
        total_withdrawals = round(sum(v.get("withdrawals", 0.0) for v in flows_by_date.values()), 2)
        total_capital_gains_tax = round(sum(v.get("capital_gains_tax", 0.0) for v in flows_by_date.values()), 2)
        net_external_flow = round(sum(v.get("net_external_flow", 0.0) for v in flows_by_date.values()), 2)

        start_dividends = state["cumulative_dividends"][start_idx]
        end_dividends = state["cumulative_dividends"][end_idx]
        net_dividends = round(end_dividends - start_dividends, 2)

        in_period_values = [v for v in state["market_values"][start_idx + 1:end_idx + 1] if v is not None]
        average_market_value = round(sum(in_period_values) / len(in_period_values), 2) if in_period_values else end_market_value
        dividend_yield = (net_dividends / average_market_value) if average_market_value and average_market_value > 0 else None

        xirr_cashflows: list[tuple[date, float]] = []
        if start_market_value > 0:
            xirr_cashflows.append((start, -start_market_value))
        for d in sorted(flows_by_date.keys()):
            item = flows_by_date[d]
            cf = 0.0
            cf -= item.get("deposits", 0.0)
            cf += item.get("withdrawals", 0.0)
            cf -= item.get("capital_gains_tax", 0.0)
            if abs(cf) > 1e-9:
                xirr_cashflows.append((d, round(cf, 2)))
        if end_market_value > 0:
            xirr_cashflows.append((end, end_market_value))
        xirr = self._solve_xirr(xirr_cashflows)

        cashflow_breakdown = []
        for d in sorted(flows_by_date.keys()):
            item = flows_by_date[d]
            cashflow_breakdown.append({
                "date": d.isoformat(),
                "deposits": item.get("deposits", 0.0),
                "withdrawals": item.get("withdrawals", 0.0),
                "capital_gains_tax": item.get("capital_gains_tax", 0.0),
                "net_external_flow": item.get("net_external_flow", 0.0),
            })

        result = {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "display_currency": display_currency,
            "has_data": True,
            "has_missing_prices": state["has_missing_prices"],
            "missing_data": state.get("missing_data", {}),
            "start_market_value": round(start_market_value, 2),
            "end_market_value": round(end_market_value, 2),
            "average_market_value": average_market_value,
            "net_dividends": net_dividends,
            "dividend_yield_pct": round(dividend_yield * 100, 4) if dividend_yield is not None else None,
            "net_deposits": total_deposits,
            "withdrawals": total_withdrawals,
            "capital_gains_tax": total_capital_gains_tax,
            "net_external_flow": net_external_flow,
            "twr_pct": round(twr * 100, 4) if twr is not None else None,
            "xirr_pct": round(xirr * 100, 4) if xirr is not None else None,
            "days": (end - start).days + 1,
            "periods_used": periods_used,
            "cashflows": cashflow_breakdown,
        }
        self._result_memo[memo_key] = result
        return result

    def get_rolling_return_timeseries(self,
                                      display_currency: str = "ILS",
                                      point_frequency: str = "W",
                                      window_days: int = 365) -> dict:
        """
        Rolling return series using precomputed daily r_t and cumulative logs.

        point_frequency controls datapoint spacing: 'W' or 'M'.
        window_days controls the rolling window (default: 365 days).
        """
        point_frequency = (point_frequency or "W").upper()
        if point_frequency not in ("W", "M"):
            point_frequency = "W"
        window_days = max(int(window_days or 365), 7)

        memo_key = ("rolling_return_timeseries", display_currency, point_frequency, window_days)
        if memo_key in self._result_memo:
            return self._result_memo[memo_key]

        start = self.start_date
        end = self.end_date
        state = self._build_return_state(start, end, display_currency)
        date_to_index = state["date_to_index"]

        anchor_dates = self._date_range(start, end, point_frequency)
        points = []

        for anchor in anchor_dates:
            window_start = anchor - timedelta(days=window_days - 1)
            if window_start < start:
                continue  # require a full rolling window

            start_idx = date_to_index.get(window_start - timedelta(days=1))
            end_idx = date_to_index.get(anchor)
            if start_idx is None or end_idx is None:
                continue

            twr = self._window_twr_from_state(state, start_idx, end_idx)
            if twr is None:
                continue

            points.append({
                "date": anchor.isoformat(),
                "window_start": window_start.isoformat(),
                "window_end": anchor.isoformat(),
                "twr_pct": round(twr * 100, 4),
            })

        result = {
            "display_currency": display_currency,
            "point_frequency": point_frequency,
            "window_days": window_days,
            "points": points,
            "has_missing_prices": state["has_missing_prices"],
            "missing_data": state.get("missing_data", {}),
        }
        self._result_memo[memo_key] = result
        return result

    def get_yearly_yield(self,
                         year: int,
                         display_currency: str = "ILS",
                         override_start_mv: float | None = None) -> dict:
        """
        Year-specific wrapper around get_portfolio_return_summary().

        override_start_mv: when provided (passed from get_all_yearly_yields),
          replaces the independently-computed start_market_value and
          recomputes TWR and XIRR with that value so they are consistent with
          the previous year's end value.
        """
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        start = max(year_start, self.start_date)
        end = min(year_end, self.end_date)

        if end < start:
            return {
                "year": year,
                "display_currency": display_currency,
                "has_data": False,
                "error": "Year is outside the portfolio date range",
            }

        result = self.get_portfolio_return_summary(start, end, display_currency)
        result["year"] = year
        result["partial_year"] = (start != year_start or end != year_end)

        if not result.get("has_data"):
            return result

        # If caller supplies the authoritative start value (previous year's end),
        # patch start_market_value and recompute TWR and XIRR from scratch so
        # all three figures are mutually consistent.
        if override_start_mv is not None and override_start_mv != result.get("start_market_value"):
            result["start_market_value"] = round(override_start_mv, 2)

            # Recompute TWR: (end - start - net_flows) / start  — simplified single-
            # period approximation is wrong for multi-deposit years, so rebuild the
            # daily TWR using the state's cumulative log but with the corrected
            # start value embedded in the XIRR only (TWR is flow-weighted, so it
            # is unaffected by the absolute start value; it remains valid as-is).
            # Only XIRR needs the corrected start value.
            end_mv = result.get("end_market_value", 0.0)
            flows_by_date = result.get("cashflows", [])

            xirr_cashflows: list[tuple[date, float]] = []
            if override_start_mv > 0:
                xirr_cashflows.append((start, -override_start_mv))
            for cf_item in flows_by_date:
                d = date.fromisoformat(cf_item["date"])
                cf = 0.0
                cf -= cf_item.get("deposits", 0.0)
                cf += cf_item.get("withdrawals", 0.0)
                cf -= cf_item.get("capital_gains_tax", 0.0)
                if abs(cf) > 1e-9:
                    xirr_cashflows.append((d, round(cf, 2)))
            if end_mv > 0:
                xirr_cashflows.append((end, end_mv))
            xirr = self._solve_xirr(xirr_cashflows)
            result["xirr_pct"] = round(xirr * 100, 4) if xirr is not None else None

        return result

    def get_all_yearly_yields(self,
                              display_currency: str = "ILS") -> dict:
        """Returns yearly yield summaries for all years in the portfolio range."""
        memo_key = ("all_yearly_yields", display_currency)
        if memo_key in self._result_memo:
            return self._result_memo[memo_key]

        years = []
        prev_end_mv: float | None = None
        for year in range(self.start_date.year, self.end_date.year + 1):
            # Pass the previous year's end value so start is exactly consistent.
            summary = self.get_yearly_yield(year, display_currency,
                                            override_start_mv=prev_end_mv)
            if summary.get("has_data"):
                years.append(summary)
                prev_end_mv = summary.get("end_market_value")

        result = {
            "display_currency": display_currency,
            "years": years,
        }
        self._result_memo[memo_key] = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # DATE RANGE HELPER
    # ─────────────────────────────────────────────────────────────────────────

    def _date_range(self, start: date, end: date, frequency: str) -> list[date]:
        freq_map = {"D": 1, "W": 7, "M": None}
        dates = []
        cur   = start
        if frequency == "M":
            while cur <= end:
                dates.append(cur)
                cur += relativedelta(months=1)
        else:
            step = timedelta(days=freq_map.get(frequency, 7))
            while cur <= end:
                dates.append(cur)
                cur += step
        if dates and dates[-1] != end:
            dates.append(end)
        return dates

