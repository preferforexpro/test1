"""
main.py  v3.4  —  SMC Forex Scanner Backend
═════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE DOES:
─────────────────────────────────────────────────────────────────────────────
  1. Connects to MetaTrader 5 via MT5Manager (or falls back to GBM sim)
  2. Runs an initial parallel scan of all 47 forex pairs on startup
  3. Keeps a rolling scan running in the background (re-analyses 3 pairs
     every 5 seconds → full cycle ≈ 80 seconds)
  4. Pushes live results + alerts to connected browsers via WebSocket
  5. Serves index.html and chart.html directly so no CORS issues

HOW ALERTS ARE GENERATED:
─────────────────────────────────────────────────────────────────────────────
  Alerts fire when the SMC engine detects a NEW signal that was absent in
  the previous scan cycle:

  ┌─────────────┬──────────────────────────────────────────────────────────┐
  │ Alert Type  │ What triggers it                                         │
  ├─────────────┼──────────────────────────────────────────────────────────┤
  │ CHoCH       │ choch was None → now 1 or -1                             │
  │             │ Price broke the LAST swing high/low against trend.       │
  │             │ Highest priority — potential reversal signal.            │
  ├─────────────┼──────────────────────────────────────────────────────────┤
  │ BOS         │ bos was None → now 1 or -1                               │
  │             │ Price broke a previous swing high (bull) or low (bear).  │
  │             │ Trend continuation confirmation.                         │
  ├─────────────┼──────────────────────────────────────────────────────────┤
  │ FVG         │ fvg was None/mitigated → new unmitigated FVG appears     │
  │             │ A 3-candle imbalance gap that price hasn't filled yet.   │
  ├─────────────┼──────────────────────────────────────────────────────────┤
  │ Bias Flip   │ bias changed direction (e.g. BULL→BEAR) and ≠ 0         │
  │             │ The composite SMC score crossed the ±1.5 threshold.      │
  └─────────────┴──────────────────────────────────────────────────────────┘

  WEEKEND SUPPRESSION: Forex markets are closed Sat 00:00–Sun 22:00 UTC.
  During this window alerts are suppressed for live MT5 pairs because no new
  candles form — any apparent "change" is just floating-point noise from
  repeated analysis of the same static historical data.
  The scanner slows from 5s → 30s intervals on weekends to save resources.

HOW THE SUMMARY TEXT IS GENERATED:
─────────────────────────────────────────────────────────────────────────────
  The plain-English summary in the Pair Drawer is assembled in index.html
  inside the calcSetup() function. It works like this:

  1. Checks which SMC signals are present in the analysis result
  2. Builds a sentence for each signal that is active:
     - CHoCH/BOS  → "CHoCH confirms bearish shift."
     - Order Block → "Bearish OB at 1.36062 is the key level."
     - FVG        → "Bear FVG gap 1.35831–1.35884 unmitigated."
     - P/D Zone   → "Price in premium zone (56.6% of range)."
     - Liquidity  → "SSL Below at 1.34579 is a liquidity target."
     - MTF align  → "MTF alignment supports bearish bias."
  3. Joins all present sentences with a space into one paragraph

  The price values (1.36062, 1.35831 etc.) come from smc_engine.py which
  reads them directly from the MT5 candle data returned by mt5_manager.py.

API ENDPOINTS:
─────────────────────────────────────────────────────────────────────────────
  GET  /                    → serves index.html (main scanner UI)
  GET  /chart               → serves chart.html (full-screen chart page)
  GET  /favicon.ico         → empty response (stops browser 404 spam)
  GET  /api/status          → full server state as JSON (debug)
  GET  /api/info            → broker/terminal info + client counts
  GET  /api/pairs           → list of all 47 supported pairs
  GET  /api/scan            → full scan results from cache
  GET  /api/candles/{sym}   → OHLCV bars  ?tf=1h&count=150
  GET  /api/analyse/{sym}   → full SMC analysis for one pair ?tf=1h
  WS   /ws/scan             → live push stream (updates + alerts)

HOW TO RUN:
─────────────────────────────────────────────────────────────────────────────
  1. Copy .env.example → .env and fill in your MT5 credentials
  2. uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  3. Open http://localhost:8000 in your browser
  4. Click any pair row → Pair Drawer opens → click "📈 Open Full Chart"
     to open the full-screen chart at http://localhost:8000/chart?sym=EURUSD
"""

import asyncio
import datetime as dt
import json
import logging
import os
import pathlib
import time

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

# ── Load .env file if present ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv not installed — use system env vars directly

# ── Internal modules ──────────────────────────────────────────────────────
from mt5_manager import MT5Manager, FOREX_PAIRS
from smc_engine   import analyse

# ═════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smc")

# ═════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (from .env or environment variables)
# ═════════════════════════════════════════════════════════════════════════
MT5_LOGIN    = int(os.getenv("MT5_LOGIN",    "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD",     "")
MT5_SERVER   = os.getenv("MT5_SERVER",       "")
MT5_PATH     = os.getenv("MT5_PATH",         "")

DEFAULT_TF   = "1h"    # default timeframe for the rolling scan
DEFAULT_CNT  = 150     # number of candles to fetch per pair

# ═════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═════════════════════════════════════════════════════════════════════════
mgr:           MT5Manager = None    # type: ignore  — set in lifespan
scan_cache:    dict       = {}      # symbol → latest SMC analysis dict
scan_clients:  set        = set()   # active WebSocket /ws/scan connections
ALL_SYMS       = list(FOREX_PAIRS.keys())

# Files are served from the same folder as main.py
_here = pathlib.Path(__file__).parent

# ═════════════════════════════════════════════════════════════════════════
#  APP LIFESPAN  (startup / shutdown)
# ═════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mgr
    log.info("Starting MT5Manager…")
    mgr = MT5Manager(
        login    = MT5_LOGIN,
        password = MT5_PASSWORD,
        server   = MT5_SERVER,
        path     = MT5_PATH,
    )
    log.info(f"Mode: {mgr.mode}  |  connected: {mgr.connected}")

    # Start background tasks
    asyncio.create_task(_initial_scan())
    asyncio.create_task(_rolling_scan())

    yield   # ← server runs here

    log.info("Shutting down MT5…")
    mgr.shutdown()


app = FastAPI(
    title       = "SMC Forex Scanner API",
    version     = "3.4.0",
    description = "MetaTrader 5 + Smart Money Concepts real-time backend",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ═════════════════════════════════════════════════════════════════════════
#  STATIC FILE SERVING
# ═════════════════════════════════════════════════════════════════════════

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Stops the browser from spamming 404 for favicon."""
    return Response(content=b"", media_type="image/x-icon")


@app.get("/", response_class=HTMLResponse)
async def serve_scanner():
    """
    Serves the main scanner UI (index.html).
    Serving via FastAPI means the browser connects via same-origin WebSocket
    ws://localhost:8000/ws/scan — no CORS issues.
    """
    p = _here / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h2>index.html not found</h2>"
        "<p>Make sure <code>index.html</code> is in the same folder as <code>main.py</code></p>",
        status_code=404,
    )


@app.get("/chart", response_class=HTMLResponse)
async def serve_chart():
    """
    Serves the full-screen SMC chart page (chart.html).
    Opened in a new browser tab from the Pair Drawer in the scanner.
    URL format: http://localhost:8000/chart?sym=EURUSD&tf=1h
    """
    p = _here / "chart.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h2>chart.html not found</h2>"
        "<p>Make sure <code>chart.html</code> is in the same folder as <code>main.py</code></p>",
        status_code=404,
    )

# ═════════════════════════════════════════════════════════════════════════
#  REST API ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def api_status():
    """
    Full server state as JSON.
    Open http://localhost:8000/api/status to verify everything is working.
    Shows mode, connected pairs, cached results, and a sample EUR/USD result.
    """
    info = mgr.terminal_info()
    return {
        "version":      "3.4.0",
        "mode":         str(mgr.mode),
        "connected":    mgr.connected,
        "is_weekend":   _is_weekend(),
        "market_open":  not _is_weekend(),
        "cached_pairs": len(scan_cache),
        "total_pairs":  len(ALL_SYMS),
        "ws_clients":   len(scan_clients),
        "terminal":     info,
        "eurusd":       scan_cache.get("EURUSD"),   # sample to verify data
    }


@app.get("/api/info")
async def api_info():
    """Broker/terminal info + connection status."""
    info = mgr.terminal_info()
    return {
        **info,
        "cached":     len(scan_cache),
        "ws_clients": len(scan_clients),
        "is_weekend": _is_weekend(),
    }


@app.get("/api/pairs")
async def api_pairs():
    """List of all supported forex pairs."""
    return {"pairs": ALL_SYMS, "count": len(ALL_SYMS)}


@app.get("/api/scan")
async def api_scan():
    """
    Returns the full cached scan result for all pairs.
    This is the HTTP fallback — the WebSocket /ws/scan is preferred for
    live updates, but this endpoint is useful for REST clients or debugging.
    """
    return {
        "count":      len(scan_cache),
        "results":    scan_cache,
        "mode":       str(mgr.mode),
        "is_weekend": _is_weekend(),
    }


@app.get("/api/candles/{symbol}")
async def api_candles(
    symbol: str,
    tf:    str = Query("1h",  description="Timeframe: 1m 5m 15m 30m 1h 4h 1d"),
    count: int = Query(150,   ge=20, le=500, description="Number of bars"),
):
    """
    OHLCV candle data for a symbol.
    Used by chart.html to draw the candlestick chart.
    Example: GET /api/candles/EURUSD?tf=1h&count=150
    """
    sym = symbol.upper()
    df  = mgr.get_candles(sym, tf, count)
    if df is None:
        raise HTTPException(status_code=404, detail=f"No data for {sym} on {tf}")
    return {
        "symbol":   sym,
        "tf":       tf,
        "count":    len(df),
        "candles":  df.to_dict(orient="records"),
    }


@app.get("/api/analyse/{symbol}")
async def api_analyse(
    symbol: str,
    tf:    str = Query("1h"),
    count: int = Query(150, ge=30, le=500),
):
    """
    Full SMC analysis for a single pair.
    Used by chart.html to get all signal levels to draw on the chart.
    Returns: price, bias, FVG, OB, BOS, CHoCH, liquidity, swing, P/D zone, score.
    Example: GET /api/analyse/EURUSD?tf=1h
    """
    sym = symbol.upper()
    df  = mgr.get_candles(sym, tf, count)
    r   = analyse(df, sym, tf)
    if r is None:
        raise HTTPException(status_code=422, detail=f"Not enough data to analyse {sym}")
    return r

# ═════════════════════════════════════════════════════════════════════════
#  WEBSOCKET  /ws/scan  —  live push stream
# ═════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/scan")
async def ws_scan(ws: WebSocket):
    """
    WebSocket endpoint that pushes SMC updates to the browser in real-time.

    Message types sent TO the browser:
    ─────────────────────────────────
    init   : Full cache snapshot sent immediately on connect, and again after
             the initial parallel scan completes.
             { type:"init", data:{EURUSD:{...}, GBPUSD:{...}, ...},
               mode:"live", cached:47 }

    update : Incremental update for one pair after it is re-analysed.
             { type:"update", symbol:"EURUSD", data:{...} }

    alert  : New SMC signal detected (CHoCH, BOS, FVG, Bias Flip).
             Suppressed on weekends for live MT5 pairs.
             { type:"alert", data:{symbol, signal, direction, price, level, ts} }

    ping   : Heartbeat every 25 seconds to keep connection alive.
             { type:"ping", ts:1714567890000, mode:"live", cached:47 }

    Messages received FROM the browser:
    ─────────────────────────────────────
    { type:"ping" }   → server replies { type:"pong", ts:... }
    { type:"resync" } → server re-sends full init immediately
    """
    await ws.accept()
    scan_clients.add(ws)
    log.info(f"WS+  clients={len(scan_clients)}  mode={mgr.mode}  cached={len(scan_cache)}")

    # Send current cache immediately — browser gets data straight away
    try:
        await ws.send_json({
            "type":   "init",
            "data":   scan_cache,
            "mode":   str(mgr.mode),
            "cached": len(scan_cache),
        })
    except Exception as e:
        log.error(f"WS init send failed: {e}")
        scan_clients.discard(ws)
        return

    # Keep connection alive and handle client messages
    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=25.0)
                try:
                    msg = json.loads(raw)
                    if msg.get("type") == "ping":
                        await ws.send_json({"type": "pong", "ts": _ms()})
                    elif msg.get("type") == "resync":
                        # Client requested full refresh
                        await ws.send_json({
                            "type":   "init",
                            "data":   scan_cache,
                            "mode":   str(mgr.mode),
                            "cached": len(scan_cache),
                        })
                except json.JSONDecodeError:
                    pass  # ignore malformed messages

            except asyncio.TimeoutError:
                # Send heartbeat ping to keep the connection alive
                try:
                    await ws.send_json({
                        "type":   "ping",
                        "ts":     _ms(),
                        "mode":   str(mgr.mode),
                        "cached": len(scan_cache),
                    })
                except Exception:
                    break  # connection is dead

    except (WebSocketDisconnect, Exception):
        pass
    finally:
        scan_clients.discard(ws)
        log.info(f"WS-  clients={len(scan_clients)}")

# ═════════════════════════════════════════════════════════════════════════
#  BACKGROUND TASK 1 — INITIAL SCAN
# ═════════════════════════════════════════════════════════════════════════

async def _initial_scan():
    """
    Runs ALL 47 pairs in parallel on startup (up to 6 concurrent MT5 calls).
    Each pair result is pushed to connected browsers the moment it's ready.
    Completes in ~5–10 seconds instead of 38 seconds (sequential would be slow).
    After all pairs are done, sends a full 'init' to refresh all browsers.
    """
    log.info(f"Initial scan: {len(ALL_SYMS)} pairs (parallel, max 6 concurrent)…")
    loop = asyncio.get_event_loop()
    sem  = asyncio.Semaphore(6)   # limit concurrent MT5 calls

    async def _one(sym: str):
        async with sem:
            try:
                df = await loop.run_in_executor(
                    None,
                    lambda s=sym: mgr.get_candles(s, DEFAULT_TF, DEFAULT_CNT)
                )
                r = analyse(df, sym, DEFAULT_TF)
                if r:
                    scan_cache[sym] = r
                    label = ["BEAR","NEUT","BULL"][r["bias"] + 1]
                    log.info(f"  {sym:10s}  {label}  conf={r['confluence']}")
                    # Push to any browsers already connected
                    if scan_clients:
                        await _bcast({"type": "update", "symbol": sym, "data": r})
                else:
                    log.warning(f"  {sym:10s}  no result (insufficient data)")
            except Exception as e:
                log.error(f"  {sym} initial scan error: {e}")

    await asyncio.gather(*[_one(s) for s in ALL_SYMS])
    log.info(f"Initial scan complete: {len(scan_cache)}/{len(ALL_SYMS)}  mode={mgr.mode}")

    # Send refreshed full init to all connected clients
    if scan_clients:
        await _bcast({
            "type":   "init",
            "data":   scan_cache,
            "mode":   str(mgr.mode),
            "cached": len(scan_cache),
        })

# ═════════════════════════════════════════════════════════════════════════
#  BACKGROUND TASK 2 — ROLLING SCAN
# ═════════════════════════════════════════════════════════════════════════

async def _rolling_scan():
    """
    Continuously re-analyses pairs in a rolling window:
    • Weekday:  3 pairs every 5 seconds  → full 47-pair cycle ≈ 80 seconds
    • Weekend:  3 pairs every 30 seconds → full cycle ≈ 8 minutes (saves resources,
                no new candles form during weekend anyway)

    For each pair:
    1. Fetches fresh candles from MT5 (or GBM sim)
    2. Runs SMC analysis
    3. Pushes 'update' message to all connected browsers
    4. Compares to previous result → fires alerts if new signals detected
       (alerts suppressed on weekends for live pairs)
    """
    # Wait until initial scan has populated some data
    while len(scan_cache) < min(5, len(ALL_SYMS)):
        await asyncio.sleep(2)
    log.info("Rolling scan started.")

    loop = asyncio.get_event_loop()
    prev = dict(scan_cache)   # snapshot for change detection
    idx  = 0

    while True:
        # Slow down on weekends — markets closed, no new data
        sleep_sec = 30 if _is_weekend() else 5
        await asyncio.sleep(sleep_sec)

        batch = [ALL_SYMS[(idx + k) % len(ALL_SYMS)] for k in range(3)]
        idx   = (idx + 3) % len(ALL_SYMS)

        for sym in batch:
            try:
                df = await loop.run_in_executor(
                    None,
                    lambda s=sym: mgr.get_candles(s, DEFAULT_TF, DEFAULT_CNT)
                )
                r = analyse(df, sym, DEFAULT_TF)
                if r is None:
                    continue

                scan_cache[sym] = r

                if scan_clients:
                    # Always push the updated analysis
                    await _bcast({"type": "update", "symbol": sym, "data": r})

                    # Only fire alerts when markets are open
                    # (suppress on weekends for live MT5 — static data = false signals)
                    markets_open = not _is_weekend() or mgr.mode == "simulation"
                    if markets_open:
                        for alert in _check_alerts(sym, prev.get(sym), r):
                            await _bcast({"type": "alert", "data": alert})

                prev[sym] = r

            except Exception as e:
                log.error(f"Rolling scan error [{sym}]: {e}")

# ═════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════

def _ms() -> int:
    """Current time as Unix milliseconds (used for alert timestamps)."""
    return int(time.time() * 1000)


def _is_weekend() -> bool:
    """
    Returns True when forex markets are closed for the weekend.

    Forex market schedule (UTC):
    • Friday   22:00 UTC — NY session closes (markets close)
    • Saturday (all day) — markets closed
    • Sunday   22:00 UTC — Sydney/Wellington open (markets reopen)

    So markets are closed from Friday 22:00 to Sunday 22:00 UTC.
    This function returns True during that window.
    """
    now = dt.datetime.now(dt.timezone.utc)
    wd  = now.weekday()   # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    hr  = now.hour

    if wd == 5:                    # All of Saturday
        return True
    if wd == 4 and hr >= 22:       # Friday after 22:00 UTC
        return True
    if wd == 6 and hr < 22:        # Sunday before 22:00 UTC
        return True
    return False


async def _bcast(payload: dict):
    """
    Broadcast a JSON message to all connected WebSocket clients.
    Uses difference_update() instead of -= to avoid Python treating
    scan_clients as a local variable (which causes UnboundLocalError).
    """
    dead = set()
    msg  = json.dumps(payload, default=str)
    for ws in list(scan_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    scan_clients.difference_update(dead)   # in-place, no rebinding


def _check_alerts(sym: str, old: dict, new: dict) -> list:
    """
    Compares two consecutive SMC analysis results for the same pair.
    Returns a list of alert dicts for any NEW signals detected.

    Signal change detection logic:
    ─────────────────────────────────────────────────────────────────────
    CHoCH  — was absent (None) → now present (1 or -1)
             Indicates a potential trend reversal. Highest priority.
             Example: choch was None, now choch = -1 → "Bearish CHoCH"

    BOS    — was absent (None) → now present (1 or -1)
             Indicates trend continuation. Lower priority than CHoCH.
             Example: bos was None, now bos = 1 → "Bullish BOS"

    FVG    — was absent/mitigated → new unmitigated FVG appeared
             A fresh 3-candle imbalance gap formed that hasn't been filled.

    Bias Flip — bias changed direction AND is not neutral (not 0)
             Example: bias was 1 (BULL) → now -1 (BEAR) → "Bias Flip"

    Only one alert per scan cycle per pair (highest priority wins).
    ─────────────────────────────────────────────────────────────────────
    """
    if not old:
        return []   # No previous data to compare against

    base = {
        "symbol":    sym,
        "timeframe": DEFAULT_TF,
        "price":     new["price"],
        "ts":        _ms(),
    }

    # CHoCH — highest priority
    if new.get("choch") is not None and old.get("choch") is None:
        return [{
            **base,
            "signal":    "CHoCH",
            "direction": new["choch"],
            "level":     new.get("choch_level"),
        }]

    # BOS
    if new.get("bos") is not None and old.get("bos") is None:
        return [{
            **base,
            "signal":    "BOS",
            "direction": new["bos"],
            "level":     new.get("bos_level"),
        }]

    # New FVG
    if new.get("fvg") and not old.get("fvg"):
        return [{
            **base,
            "signal":    "FVG",
            "direction": new["fvg"]["type"],
        }]

    # Bias direction changed
    if new.get("bias") != old.get("bias") and new.get("bias") != 0:
        return [{
            **base,
            "signal":    "Bias Flip",
            "direction": new["bias"],
        }]

    return []
