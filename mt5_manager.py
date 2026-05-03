"""
mt5_manager.py
──────────────
Handles MT5 connection + symbol name auto-detection per broker + GBM fallback.

Dukascopy / other brokers may suffix symbols: EURUSD, EURUSD., EURUSDm, EUR/USD
This manager probes the terminal and builds a mapping automatically.
"""

import time, threading, logging
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("mt5")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not installed — full simulation mode")

TF_MAP_STR = {
    "1m":"M1","5m":"M5","15m":"M15","30m":"M30",
    "1h":"H1","4h":"H4","1d":"D1","1w":"W1",
}

# ── Canonical pair list ──────────────────────────────────────────────────────
FOREX_PAIRS = {
     "BTCUSD":(65000.0, 0.04),"EURUSD":(1.0850,0.0060),"GBPUSD":(1.2730,0.0090),
     "USDJPY":(149.20,0.75),
    "USDCHF":(0.8980,0.0055),"AUDUSD":(0.6390,0.0065),"NZDUSD":(0.5940,0.0070),
    "USDCAD":(1.3620,0.0060),"EURGBP":(0.8550,0.0045),"EURJPY":(161.80,0.80),
    "EURCHF":(0.9720,0.0050),"EURAUD":(1.6980,0.0100),"EURNZD":(1.8250,0.0110),
    "EURCAD":(1.4780,0.0085),"GBPJPY":(189.60,1.10),"GBPCHF":(1.1410,0.0080),
    "GBPAUD":(1.9920,0.0120),"GBPNZD":(2.1420,0.0130),"GBPCAD":(1.7330,0.0100),
    "AUDJPY":(95.30,0.50),"AUDCHF":(0.5750,0.0060),"AUDNZD":(1.0750,0.0065),
    "AUDCAD":(0.8700,0.0070),"NZDJPY":(88.60,0.50),"NZDCHF":(0.5350,0.0065),
    "NZDCAD":(0.8090,0.0080),"CHFJPY":(166.20,0.80),"CADJPY":(109.50,0.60),
    "CADCHF":(0.6590,0.0055),"USDMXN":(17.15,0.15),"USDZAR":(18.60,0.18),
    "USDTRY":(32.40,0.25),"USDSGD":(1.3440,0.0045),"USDHKD":(7.820,0.0020),
    "USDNOK":(10.68,0.08),"USDSEK":(10.48,0.08),"USDDKK":(6.910,0.0045),
    "USDPLN":(3.960,0.030),"USDHUF":(361.0,1.50),"USDCZK":(22.85,0.12),
    "EURTRY":(35.10,0.30),"USDINR":(83.60,0.15),"USDTHB":(35.50,0.08),
    "USDMYR":(4.710,0.015),"USDIDR":(15820.,50.),"USDPHP":(56.50,0.15),
    "USDCNH":(7.250,0.012),
}

TF_SECS = {"1m":60,"5m":300,"15m":900,"30m":1800,"1h":3600,"4h":14400,"1d":86400}


class MT5Manager:
    def __init__(self, login=0, password="", server="", path=""):
        self.login, self.password, self.server, self.path = login, password, server, path
        self._lock      = threading.Lock()
        self._connected = False
        self._sym_map   = {}   # canonical → actual broker symbol name
        self._sims      = {}   # fallback GBM simulators per symbol
        self._mode      = "disconnected"

        if not MT5_AVAILABLE:
            logger.info("MT5 package not available → simulation mode")
            self._mode = "simulation"
            self._init_all_sims()
            return

        if self._connect():
            self._build_symbol_map()
            self._mode = "live"
            # Start watchdog
            t = threading.Thread(target=self._watchdog, daemon=True)
            t.start()
        else:
            logger.warning("MT5 connect failed → simulation mode")
            self._mode = "simulation"
            self._init_all_sims()

    # ── Connection ───────────────────────────────────────────────────────────
    def _connect(self) -> bool:
        with self._lock:
            try:
                kw = {}
                if self.path:     kw["path"]     = self.path
                if self.login:    kw["login"]    = self.login
                if self.password: kw["password"] = self.password
                if self.server:   kw["server"]   = self.server
                if not mt5.initialize(**kw):
                    logger.error(f"MT5 init error: {mt5.last_error()}")
                    return False
                info = mt5.terminal_info()
                logger.info(f"MT5 connected → {info.company} / {info.name}")
                self._connected = True
                return True
            except Exception as e:
                logger.error(f"MT5 connect exception: {e}")
                return False

    def _watchdog(self):
        while True:
            time.sleep(30)
            if not self._connected:
                logger.info("Watchdog: reconnecting…")
                if self._connect():
                    self._build_symbol_map()

    # ── Symbol name auto-detection ───────────────────────────────────────────
    def _build_symbol_map(self):
        """
        Probe the broker terminal to find the actual symbol name for each
        canonical pair. Tries: EURUSD, EURUSD., EURUSDm, EUR/USD, eurusd, etc.
        Falls back to simulation for symbols not found.
        """
        with self._lock:
            all_broker_syms = {s.name for s in (mt5.symbols_get() or [])}

        logger.info(f"Broker has {len(all_broker_syms)} symbols. Mapping forex pairs…")
        mapped, missing = 0, 0

        for canon in FOREX_PAIRS:
            found = None
            # Try common variants
            for variant in [
                canon,
                canon + ".",
                canon + "m",
                canon + "+",
                canon[:3] + "/" + canon[3:],
                canon.lower(),
                canon[:3] + "_" + canon[3:],
            ]:
                if variant in all_broker_syms:
                    found = variant
                    break

            if found:
                self._sym_map[canon] = found
                # Enable symbol in Market Watch
                try:
                    mt5.symbol_select(found, True)
                except Exception:
                    pass
                mapped += 1
            else:
                self._sym_map[canon] = None   # will use simulation
                self._sims[canon]    = _SymSim(canon, *FOREX_PAIRS[canon])
                missing += 1

        logger.info(f"Symbol map: {mapped} live, {missing} simulation fallback")

    # ── Public API ───────────────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def connected(self) -> bool:
        return self._connected or self._mode == "simulation"

    def get_candles(self, canonical: str, timeframe: str = "1h", count: int = 150) -> Optional[pd.DataFrame]:
        """Returns DataFrame[time,open,high,low,close,volume] or None."""
        # Simulation mode for this symbol?
        if self._mode == "simulation" or self._sym_map.get(canonical) is None:
            sim = self._sims.get(canonical)
            if sim is None:
                bp, dv = FOREX_PAIRS.get(canonical, (1.0, 0.001))
                self._sims[canonical] = _SymSim(canonical, bp, dv)
                sim = self._sims[canonical]
            rows = sim.get_candles(timeframe, count)
            df   = pd.DataFrame(rows)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df

        # Live MT5
        broker_sym = self._sym_map[canonical]
        tf_const   = _tf_const(timeframe)
        if tf_const is None:
            return None

        with self._lock:
            rates = mt5.copy_rates_from_pos(broker_sym, tf_const, 0, count)

        if rates is None or len(rates) == 0:
            logger.warning(f"No MT5 data for {broker_sym} {timeframe} → sim fallback")
            sim = self._sims.get(canonical)
            if sim is None:
                bp, dv = FOREX_PAIRS.get(canonical, (1.0, 0.001))
                self._sims[canonical] = _SymSim(canonical, bp, dv)
                sim = self._sims[canonical]
            rows = sim.get_candles(timeframe, count)
            df   = pd.DataFrame(rows)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(columns={"tick_volume":"volume"}, inplace=True)
        return df[["time","open","high","low","close","volume"]].copy()

    def get_tick(self, canonical: str) -> Optional[dict]:
        if self._mode == "simulation" or self._sym_map.get(canonical) is None:
            sim = self._sims.get(canonical)
            return sim.tick() if sim else None

        broker_sym = self._sym_map[canonical]
        with self._lock:
            t = mt5.symbol_info_tick(broker_sym)
        if t is None:
            return None
        mid = (t.bid + t.ask) / 2
        return {
            "symbol": canonical,
            "time":   datetime.fromtimestamp(t.time, tz=timezone.utc).isoformat(),
            "bid":    round(t.bid, 6),
            "ask":    round(t.ask, 6),
            "last":   round(mid, 6),
            "spread": round((t.ask - t.bid) * 1e5, 1),
        }

    def terminal_info(self) -> dict:
        if not MT5_AVAILABLE or self._mode == "simulation":
            return {"mode":"simulation","connected":True,
                    "company":"Simulation","server":"GBM Engine"}
        with self._lock:
            info = mt5.terminal_info()
            acct = mt5.account_info()
        return {
            "mode":     "live",
            "connected": self._connected,
            "company":  getattr(acct,  "company", "—"),
            "server":   getattr(acct,  "server",  "—"),
            "terminal": getattr(info,  "name",    "—"),
            "login":    getattr(acct,  "login",   0),
            "live_pairs": len([v for v in self._sym_map.values() if v]),
            "sim_pairs":  len([v for v in self._sym_map.values() if not v]),
        }

    def get_symbols(self) -> list:
        return list(FOREX_PAIRS.keys())

    def shutdown(self):
        if MT5_AVAILABLE and self._mode == "live":
            with self._lock:
                mt5.shutdown()
            logger.info("MT5 shutdown.")

    # ── Init all sims ────────────────────────────────────────────────────────
    def _init_all_sims(self):
        for sym, (bp, dv) in FOREX_PAIRS.items():
            self._sims[sym] = _SymSim(sym, bp, dv)


# ─── Timeframe helper ────────────────────────────────────────────────────────
def _tf_const(tf: str):
    if not MT5_AVAILABLE:
        return None
    MAP = {
        "1m":  mt5.TIMEFRAME_M1,  "5m":  mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30,
        "1h":  mt5.TIMEFRAME_H1,  "4h":  mt5.TIMEFRAME_H4,
        "1d":  mt5.TIMEFRAME_D1,  "1w":  mt5.TIMEFRAME_W1,
    }
    return MAP.get(tf)


# ─── GBM Simulator ───────────────────────────────────────────────────────────
class _SymSim:
    _rng = np.random.default_rng()

    def __init__(self, symbol: str, base_price: float, daily_vol: float):
        self.symbol    = symbol
        self.price     = base_price
        self.daily_vol = daily_vol
        self.trend     = 0.0
        self.trend_rem = 0
        self._cache    = {}   # tf → list of candle dicts (regenerated each call)

    def _bar_vol(self, tf: str) -> float:
        secs = TF_SECS.get(tf, 3600)
        return self.daily_vol * (secs / 86400) ** 0.5

    def _step_trend(self):
        if self.trend_rem <= 0:
            r = float(self._rng.random())
            self.trend    = 1.0 if r < 0.35 else (-1.0 if r < 0.70 else 0.0)
            self.trend_rem = int(self._rng.integers(8, 45))
        self.trend_rem -= 1

    def get_candles(self, tf: str, count: int) -> list:
        """Generate `count` synthetic OHLCV bars ending now."""
        secs  = TF_SECS.get(tf, 3600)
        bvol  = self._bar_vol(tf)
        price = self.price
        now   = int(time.time())
        bars  = []
        for i in range(count, 0, -1):
            self._step_trend()
            move  = self.trend * bvol * 0.2 + float(self._rng.normal(0, bvol))
            o     = price
            c     = max(o * 0.0001, o * (1 + move))
            rng_  = abs(move) + float(self._rng.exponential(bvol * 0.5))
            h     = max(o, c) * (1 + abs(float(self._rng.exponential(rng_ * 0.4))))
            l     = min(o, c) * (1 - abs(float(self._rng.exponential(rng_ * 0.4))))
            ts    = now - i * secs
            bars.append({
                "time":   datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "open":   round(o, 6), "high": round(h, 6),
                "low":    round(l, 6), "close": round(c, 6),
                "volume": int(self._rng.integers(10_000, 500_000)),
            })
            price = c
        self.price = price   # update running price
        return bars

    def tick(self) -> dict:
        self._step_trend()
        bvol  = self._bar_vol("1m")
        micro = self.trend * bvol * 0.015 + float(self._rng.normal(0, bvol * 0.1))
        self.price = max(self.price * 0.0001, self.price * (1 + micro))
        return {
            "symbol": self.symbol,
            "bid":    round(self.price * 0.9999, 6),
            "ask":    round(self.price * 1.0001, 6),
            "last":   round(self.price, 6),
            "spread": 1.0,
            "time":   datetime.now(tz=timezone.utc).isoformat(),
        }
