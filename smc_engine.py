"""
smc_engine.py — Smart Money Concepts analysis
Root fix: every numpy.bool_ / numpy.float64 / numpy.int_ converted to
          native Python bool / float / int before returning.
"""
from __future__ import annotations
from typing import Optional, Any
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  NUMPY → PYTHON TYPE SANITIZER
#  Call _clean(obj) on any dict/list before returning to FastAPI or json.dumps
# ─────────────────────────────────────────────────────────────────────────────
def _clean(obj: Any) -> Any:
    """Recursively convert numpy scalars to native Python types."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
#  SWING HIGHS / LOWS
# ─────────────────────────────────────────────────────────────────────────────
def find_swings(df: pd.DataFrame, length: int = 5) -> list:
    h, l, n = df["high"].values, df["low"].values, len(df)
    out = []
    for i in range(length, n - length):
        if all(h[i] >= h[j] for j in range(i - length, i + length + 1) if j != i):
            out.append({"index": int(i), "type": 1,  "level": float(h[i])})
        elif all(l[i] <= l[j] for j in range(i - length, i + length + 1) if j != i):
            out.append({"index": int(i), "type": -1, "level": float(l[i])})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  FAIR VALUE GAPS
# ─────────────────────────────────────────────────────────────────────────────
def find_fvg(df: pd.DataFrame) -> list:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    last = float(c[-1])
    out  = []
    for i in range(1, len(df) - 1):
        if float(h[i-1]) < float(l[i+1]):          # bullish gap
            bot = float(h[i-1])
            top = float(l[i+1])
            out.append({
                "type": 1, "index": int(i),
                "top": top, "bot": bot,
                "mid": round((top + bot) / 2, 6),
                "mitigated": bool(last < bot),      # ← native bool
            })
        elif float(l[i-1]) > float(h[i+1]):         # bearish gap
            bot = float(h[i+1])
            top = float(l[i-1])
            out.append({
                "type": -1, "index": int(i),
                "top": top, "bot": bot,
                "mid": round((top + bot) / 2, 6),
                "mitigated": bool(last > top),      # ← native bool
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  ORDER BLOCKS
# ─────────────────────────────────────────────────────────────────────────────
def find_order_blocks(df: pd.DataFrame, swings: list, lookback: int = 20) -> list:
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    last = float(c[-1])
    obs  = []

    hi_swings = [s for s in swings if s["type"] ==  1]
    lo_swings = [s for s in swings if s["type"] == -1]

    for sw in hi_swings[-5:]:
        for k in range(sw["index"] - 1, max(0, sw["index"] - lookback) - 1, -1):
            if float(c[k]) < float(o[k]):                  # bearish candle → bullish OB
                obs.append({
                    "type": 1, "index": int(k),
                    "top": float(h[k]), "bot": float(c[k]),
                    "mitigated": bool(last < float(c[k])),  # ← native bool
                })
                break

    for sw in lo_swings[-5:]:
        for k in range(sw["index"] - 1, max(0, sw["index"] - lookback) - 1, -1):
            if float(c[k]) > float(o[k]):                  # bullish candle → bearish OB
                obs.append({
                    "type": -1, "index": int(k),
                    "top": float(c[k]), "bot": float(l[k]),
                    "mitigated": bool(last > float(c[k])),  # ← native bool
                })
                break

    return obs


# ─────────────────────────────────────────────────────────────────────────────
#  BOS / CHoCH
# ─────────────────────────────────────────────────────────────────────────────
def find_bos_choch(df: pd.DataFrame, swings: list) -> dict:
    hi = [s for s in swings if s["type"] ==  1]
    lo = [s for s in swings if s["type"] == -1]
    R  = {"bos": None, "choch": None, "bos_level": None, "choch_level": None}
    if len(hi) < 2 or len(lo) < 2:
        return R

    rH, pH = hi[-1], hi[-2]
    rL, pL = lo[-1], lo[-2]
    bull_str = rH["level"] > pH["level"] and rL["level"] > pL["level"]
    bear_str = rH["level"] < pH["level"] and rL["level"] < pL["level"]

    rec = [float(v) for v in df["close"].values[-15:]]
    hb  = any(v > rH["level"] for v in rec)
    lb  = any(v < rL["level"] for v in rec)

    if hb:
        if bull_str: R["bos"] = 1;  R["bos_level"]   = float(rH["level"])
        else:        R["choch"] = 1; R["choch_level"] = float(rH["level"])
    elif lb:
        if bear_str: R["bos"] = -1;  R["bos_level"]   = float(rL["level"])
        else:        R["choch"] = -1; R["choch_level"] = float(rL["level"])
    return R


# ─────────────────────────────────────────────────────────────────────────────
#  LIQUIDITY POOLS
# ─────────────────────────────────────────────────────────────────────────────
def find_liquidity(df: pd.DataFrame, swings: list, tol: float = 0.0025) -> Optional[dict]:
    hi   = [s for s in swings if s["type"] ==  1]
    lo   = [s for s in swings if s["type"] == -1]
    last = float(df["close"].values[-1])
    bsl = ssl = None

    for i in range(len(hi) - 1):
        for j in range(i + 1, len(hi)):
            if abs(hi[i]["level"] - hi[j]["level"]) / hi[i]["level"] < tol:
                lvl = (hi[i]["level"] + hi[j]["level"]) / 2
                bsl = {"level": float(lvl),
                       "swept": bool(last > max(hi[i]["level"], hi[j]["level"]))}
                break
        if bsl: break

    for i in range(len(lo) - 1):
        for j in range(i + 1, len(lo)):
            if abs(lo[i]["level"] - lo[j]["level"]) / lo[i]["level"] < tol:
                lvl = (lo[i]["level"] + lo[j]["level"]) / 2
                ssl = {"level": float(lvl),
                       "swept": bool(last < min(lo[i]["level"], lo[j]["level"]))}
                break
        if ssl: break

    if bsl and not bsl["swept"]: return {"type": 1,  "label": "BSL Above", "level": bsl["level"]}
    if ssl and not ssl["swept"]: return {"type": -1, "label": "SSL Below", "level": ssl["level"]}
    if bsl and bsl["swept"]:     return {"type": -1, "label": "BSL Swept", "level": bsl["level"]}
    if ssl and ssl["swept"]:     return {"type": 1,  "label": "SSL Swept", "level": ssl["level"]}
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  SWING STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
def find_swing_structure(swings: list) -> Optional[dict]:
    hi = [s for s in swings if s["type"] ==  1]
    lo = [s for s in swings if s["type"] == -1]
    if len(hi) < 2 or len(lo) < 2:
        return None
    hh = hi[-1]["level"] > hi[-2]["level"]
    hl = lo[-1]["level"] > lo[-2]["level"]
    lh = hi[-1]["level"] < hi[-2]["level"]
    ll = lo[-1]["level"] < lo[-2]["level"]
    if hh and hl:  return {"label": "HH / HL", "bias":  1}
    if lh and ll:  return {"label": "LH / LL", "bias": -1}
    if hh and ll:  return {"label": "HH / LL", "bias":  0}
    if lh and hl:  return {"label": "LH / HL", "bias":  0}
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  PREMIUM / DISCOUNT ZONE
# ─────────────────────────────────────────────────────────────────────────────
def find_pd_zone(df: pd.DataFrame, swings: list) -> Optional[dict]:
    hi = [s for s in swings if s["type"] ==  1]
    lo = [s for s in swings if s["type"] == -1]
    if not hi or not lo:
        return None
    top  = float(max(s["level"] for s in hi[-3:]))
    bot  = float(min(s["level"] for s in lo[-3:]))
    mid  = (top + bot) / 2
    last = float(df["close"].values[-1])
    rng  = top - bot
    return {
        "top":  round(top, 6),
        "bot":  round(bot, 6),
        "mid":  round(mid, 6),
        "zone": "premium" if last > mid else "discount",
        "pct":  round((last - bot) / rng * 100, 1) if rng > 0 else 50.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN  — full SMC analysis, returns JSON-safe dict
# ─────────────────────────────────────────────────────────────────────────────
def analyse(df: Optional[pd.DataFrame], symbol: str = "", timeframe: str = "") -> Optional[dict]:
    if df is None or len(df) < 30:
        return None

    closes = df["close"].values

    sw   = find_swings(df, length=5)
    fvgs = find_fvg(df)
    obs  = find_order_blocks(df, sw)
    bos  = find_bos_choch(df, sw)
    liq  = find_liquidity(df, sw)
    str_ = find_swing_structure(sw)
    pd_  = find_pd_zone(df, sw)

    last_fvg = next((f for f in reversed(fvgs) if not f["mitigated"]), None)
    last_ob  = next((o for o in reversed(obs)  if not o["mitigated"]), None)

    last = float(closes[-1])
    prev = float(closes[-2]) if len(closes) > 1 else last
    chg  = ((last - float(closes[0])) / float(closes[0])) * 100

    # Composite score
    score = 0.0
    if last_fvg: score += last_fvg["type"] * 1.0
    if last_ob:  score += last_ob["type"]  * 1.0
    if bos["bos"]:   score += bos["bos"]   * 1.0
    if bos["choch"]: score += bos["choch"] * 1.5
    if str_:     score += str_["bias"]     * 0.8
    if liq:      score += liq["type"]      * 0.5

    bias       = 1 if score > 1.5 else (-1 if score < -1.5 else 0)
    confluence = min(6, round(abs(score) * 1.2))

    result = {
        "symbol":      symbol,
        "timeframe":   timeframe,
        "price":       round(last, 6),
        "prev_price":  round(prev, 6),
        "change_pct":  round(chg, 4),
        "tick_dir":    1 if last >= prev else -1,
        "bid":         round(last * 0.9999, 6),
        "ask":         round(last * 1.0001, 6),

        "fvg":         last_fvg,
        "ob":          last_ob,
        "bos":         bos["bos"],
        "choch":       bos["choch"],
        "bos_level":   bos["bos_level"],
        "choch_level": bos["choch_level"],
        "liquidity":   liq,
        "swing":       str_,
        "pd_zone":     pd_,

        "score":       round(float(score), 2),
        "confluence":  int(confluence),
        "bias":        int(bias),

        "spark":       [round(float(v), 6) for v in closes[-30:]],
        "candle_count": int(len(df)),
    }

    # Final safety pass — convert any remaining numpy types to Python native
    return _clean(result)
