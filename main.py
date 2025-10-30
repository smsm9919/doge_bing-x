# -*- coding: utf-8 -*-
"""
DOGE/USDT — Smart Council Pro — AI Trading System
+ Enhanced Council with MACD/VWAP/Delta Volume
+ Smart Profit Taking & Strict Exit
+ True Market Structure Detection

Exchange: BingX USDT Perp via CCXT
"""

import os, time, math, random, signal, sys, traceback, logging, json, tempfile
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from collections import deque, defaultdict

import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify, request

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# ===== ENHANCED CONFIGURATION =====
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)
PORT       = int(os.getenv("PORT", 5000))
SELF_URL   = (os.getenv("SELF_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").strip().rstrip("/")

# ===== ENHANCED STRATEGY PARAMETERS =====
SYMBOL        = "DOGE/USDT:USDT"
INTERVAL      = "15m"
LEVERAGE      = 10
RISK_ALLOC    = 0.60
POSITION_MODE = "oneway"

# Enhanced RF
RF_SOURCE     = "close"
RF_PERIOD     = 20
RF_MULT       = 3.5
RF_HYST_BPS   = 6.0

# Enhanced Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14
VWAP_LEN = 20
DELTA_VOLUME_LOOKBACK = 14

# Enhanced MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG  = 9

# Enhanced Impulse Detection
IMPULSE_ADX_MIN = 25.0
IMPULSE_BODY_ATR = 1.8
IMPULSE_MARUBOZU = 0.75
IMPULSE_VEI = 2.6
IMPULSE_VOLUME_RATIO = 2.0

# Enhanced Gates
MAX_SPREAD_BPS      = 8.0
HARD_SPREAD_BPS     = 15.0
PAUSE_ADX_THRESHOLD = 17.0

# ENHANCED COUNCIL THRESHOLDS
ENTRY_VOTES_MIN = 7  # Increased for stricter entry
ENTRY_SCORE_MIN = 5.0
ENTRY_ADX_MIN   = 22.0
EXIT_VOTES_MIN  = 4

# ENHANCED VOTING WEIGHTS - SMART COUNCIL
VOTE_SUPPLY_REJECT = 2
VOTE_DEMAND_REJECT = 2
VOTE_SWEEP         = 2
VOTE_FVG           = 1
VOTE_EQ_LEVELS     = 1
VOTE_RF_CONFIRM    = 1
VOTE_DI_ADX        = 1
VOTE_RSI_NEUT_TURN = 1
VOTE_BOOKMAP_ACC   = 1
VOTE_BOOKMAP_SWEEP = 1
VOTE_MACD_MOMENTUM = 2
VOTE_CANDLE_POWER  = 1
VOTE_IMPULSE_BONUS = 2
VOTE_TRUE_PIVOT_STRONG = 3
VOTE_TRUE_PIVOT_WEAK   = 1

# NEW ENHANCED VOTES
VOTE_VWAP_TREND    = 2
VOTE_DELTA_VOLUME  = 2
VOTE_MACD_CROSS    = 2
VOTE_VOLUME_SURGE  = 1
VOTE_TREND_ALIGN   = 1

# Enhanced Execution
MAX_SLIP_OPEN_BPS  = 20.0
MAX_SLIP_CLOSE_BPS = 30.0
USE_LIMIT_IOC      = True

# Enhanced Management
TP1_PCT_BASE       = 0.40
TP1_CLOSE_FRAC     = 0.40
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6
RATCHET_LOCK_FALLBACK = 0.60

# Enhanced Exit Conditions
EXH_MIN_PROFIT   = 0.35
OPP_RF_HYST_BPS  = 8.0
OPP_STRONG_DEBOUNCE = 2

# Enhanced Protection
STATE_FILE = "state_doge_enhanced.json"
FLIP_COOLDOWN_S = 45
MAX_TRADES_PER_HOUR = 6
CLOSE_COOLDOWN_S = 90

# ===== ENHANCED LOGGING =====
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h,"baseFilename","").endswith("bot_enhanced.log") for h in logger.handlers):
        fh = RotatingFileHandler("bot_enhanced.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(colored("🗂️ Enhanced log rotation ready","cyan"))
setup_file_logging()

# ===== ENHANCED BOOKMAP ADAPTER =====
class EnhancedBookmapAdapter:
    def __init__(self): 
        self.snapshot = []
        self.history = deque(maxlen=100)
    
    def supply(self, levels): 
        self.snapshot = levels or []
        self.history.append({
            'timestamp': time.time(),
            'levels': levels[:20] if levels else []
        })
    
    def evaluate(self, pip: float = 0.0005):
        if not self.snapshot: 
            return {"accumulation": [], "sweep": [], "walls": [], "imbalance": 0.0}
        
        by_bucket = {}
        for p, liq, imb, ab in self.snapshot:
            key = round(p / pip)
            by_bucket.setdefault(key, []).append((p, liq, imb, ab))
        
        liqs = [r[1] for r in self.snapshot if r[1] is not None]
        imbs = [r[2] for r in self.snapshot if r[2] is not None]
        liq_avg = max(1e-9, sum(liqs)/max(len(liqs),1))
        imb_avg = (sum(imbs)/max(len(imbs),1)) if imbs else 0.0
        
        zones_acc, zones_walls, zones_sweep = [], [], []
        total_imbalance = 0.0
        
        for rows in by_bucket.values():
            prices = [r[0] for r in rows]
            lo, hi = min(prices), max(prices)
            liq_sum = sum(r[1] for r in rows)
            imb_mean = sum(r[2] for r in rows)/max(len(rows),1)
            ab_hits = sum(1 for r in rows if r[3])
            
            if liq_sum > 5 * liq_avg: 
                zones_acc.append((lo, hi, liq_sum))
            if abs(imb_mean) > 2 * abs(imb_avg): 
                zones_walls.append((lo, hi, imb_mean))
            if ab_hits >= 3: 
                zones_sweep.append((lo, hi, ab_hits))
            
            total_imbalance += imb_mean * liq_sum
        
        return {
            "accumulation": zones_acc, 
            "sweep": zones_sweep, 
            "walls": zones_walls,
            "imbalance": total_imbalance
        }

bookmap = EnhancedBookmapAdapter()

# ===== ENHANCED EXCHANGE SETUP =====
def make_ex():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"}
    })

ex = make_ex()
MARKET = {}; AMT_PREC = 0; LOT_STEP = None; LOT_MIN = None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision",{}) or {}).get("amount", 0) or 0)
        lims = (MARKET.get("limits",{}) or {}).get("amount",{}) or {}
        LOT_STEP = lims.get("step"); LOT_MIN = lims.get("min")
        print(colored(f"🔧 Enhanced precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}","yellow"))

def ensure_leverage_mode():
    try:
        ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
        print(colored(f"✅ Enhanced leverage set {LEVERAGE}x","green"))
    except Exception as e:
        print(colored(f"⚠️ set_leverage warn: {e}","yellow"))

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}","yellow"))

# ===== ENHANCED HELPERS =====
def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = int(AMT_PREC) if AMT_PREC>=0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d<Decimal(str(LOT_MIN)): 
            return 0.0
        return float(d)
    except Exception: 
        return max(0.0, float(q))

def safe_qty(q):
    q = _round_amt(q)
    if q <= 0: 
        print(colored(f"⚠️ qty invalid after normalize → {q}","yellow"))
    return q

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isinf(v) or math.isnan(v))): 
            return na
        return f"{float(v):.{d}f}"
    except Exception: 
        return na

def with_retry(fn, tries=3, base=0.4):
    for i in range(tries):
        try: 
            return fn()
        except Exception:
            if i == tries-1: 
                raise
            time.sleep(base*(2**i) + random.random()*0.25)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: 
        return None

def balance_usdt():
    if not MODE_LIVE: 
        return 1000.0  # Higher paper balance for testing
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: 
        return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): 
            return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception: 
        return None

# ===== ENHANCED INDICATORS =====
def compute_enhanced_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN, VWAP_LEN, DELTA_VOLUME_LOOKBACK) + 10:
        return {
            "rsi": 50.0, "plus_di": 0.0, "minus_di": 0.0, "dx": 0.0, "adx": 0.0, 
            "atr": 0.0, "vei": 1.0, "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
            "vwap": 0.0, "delta_volume": 0.0, "volume_ratio": 1.0, "trend_strength": 0.0
        }
    
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    
    # ATR
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/ATR_LEN, adjust=False).mean()
    
    # RSI
    delta = c.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    rs = up.ewm(alpha=1/RSI_LEN, adjust=False).mean() / dn.ewm(alpha=1/RSI_LEN, adjust=False).mean().replace(0,1e-12)
    rsi = 100 - (100/(1+rs))
    
    # ADX
    upm = h.diff()
    dnm = l.shift(1)-l
    plus_dm = upm.where((upm>dnm)&(upm>0),0.0)
    minus_dm = dnm.where((dnm>upm)&(dnm>0),0.0)
    plus_di = 100*(plus_dm.ewm(alpha=1/ADX_LEN, adjust=False).mean()/atr.replace(0,1e-12))
    minus_di = 100*(minus_dm.ewm(alpha=1/ADX_LEN, adjust=False).mean()/atr.replace(0,1e-12))
    dx = (100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx = dx.ewm(alpha=1/ADX_LEN, adjust=False).mean()
    
    # VEI
    rng = (h-l).astype(float)
    try:
        lb = rng.rolling(20).mean()
        vei = (rng / lb.replace(0,1e-9))
        vei = float(vei.iloc[-1])
        if math.isinf(vei) or math.isnan(vei): 
            vei = 1.0
    except Exception:
        vei = 1.0
    
    # MACD
    ema_fast = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_sig = macd_line.ewm(span=MACD_SIG, adjust=False).mean()
    macd_hist = macd_line - macd_sig
    
    # VWAP
    typical_price = (h + l + c) / 3
    cumulative_vp = (typical_price * v).cumsum()
    cumulative_volume = v.cumsum()
    vwap = cumulative_vp / cumulative_volume.replace(0,1e-9)
    
    # Delta Volume
    price_change = c.diff()
    volume_positive = v.where(price_change > 0, 0)
    volume_negative = v.where(price_change < 0, 0)
    delta_volume = (volume_positive - volume_negative).rolling(DELTA_VOLUME_LOOKBACK).sum()
    
    # Volume Ratio
    volume_ma = v.rolling(20).mean()
    volume_ratio = v / volume_ma.replace(0,1e-9)
    
    # Trend Strength
    trend_strength = adx * (vei / 2.0)  # Combined strength indicator
    
    i = len(df)-1
    return {
        "rsi": float(rsi.iloc[i]),
        "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]),
        "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]),
        "atr": float(atr.iloc[i]),
        "vei": vei,
        "macd": float(macd_line.iloc[i]),
        "macd_signal": float(macd_sig.iloc[i]),
        "macd_hist": float(macd_hist.iloc[i]),
        "vwap": float(vwap.iloc[i]),
        "delta_volume": float(delta_volume.iloc[i]),
        "volume_ratio": float(volume_ratio.iloc[i]),
        "trend_strength": float(trend_strength.iloc[i])
    }

# ===== ENHANCED RF =====
def _ema(s: pd.Series, n: int): 
    return s.ewm(span=n, adjust=False).mean()

def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src-src.shift(1)).abs(), n)
    wper = (n*2)-1
    return _ema(avrng, wper)*qty

def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf = [float(src.iloc[0])]
    for i in range(1, len(src)):
        prev = rf[-1]
        x = float(src.iloc[i])
        r = float(rsize.iloc[i])
        cur = prev
        if x - r > prev: 
            cur = x - r
        if x + r < prev: 
            cur = x + r
        rf.append(cur)
    filt = pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def rf_signal_enhanced(df: pd.DataFrame):
    if len(df) < RF_PERIOD + 4:
        return {
            "time": int(time.time()*1000), "price": None, "long": False, "short": False,
            "filter": None, "hi": None, "lo": None, "strength": 0.0
        }
    
    d = df.iloc[:-1]
    src = d[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    
    p_now = float(src.iloc[-1])
    p_prev = float(src.iloc[-2])
    f_now = float(filt.iloc[-1])
    f_prev = float(filt.iloc[-2])
    
    def _bps(a, b):
        try: 
            return abs((a-b)/b)*10000.0
        except Exception: 
            return 0.0
    
    long_flip = (p_prev <= f_prev and p_now > f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    short_flip = (p_prev >= f_prev and p_now < f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    
    # Calculate signal strength
    strength = 0.0
    if long_flip or short_flip:
        strength = min(_bps(p_now, f_now) / RF_HYST_BPS, 2.0)  # Normalized strength
    
    return {
        "time": int(d["time"].iloc[-1]), 
        "price": p_now, 
        "long": bool(long_flip),
        "short": bool(short_flip), 
        "filter": f_now, 
        "hi": float(hi.iloc[-1]), 
        "lo": float(lo.iloc[-1]),
        "strength": strength
    }

# ===== ENHANCED MARKET STRUCTURE DETECTION =====
def detect_enhanced_pivots(df: pd.DataFrame, left: int = 3, right: int = 3):
    """Enhanced pivot detection with confirmation"""
    if len(df) < left + right + 3: 
        return None, None, None
    
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    
    ph = [None] * len(df)
    pl = [None] * len(df)
    pivot_strength = [0] * len(df)
    
    for i in range(left, len(df)-right):
        # Higher High / Lower Low confirmation
        if all(h[i] >= h[j] for j in range(i-left, i+right+1)):
            ph[i] = h[i]
            # Strength based on how much higher and volume
            strength = min((h[i] - max(h[i-left:i])) / h[i] * 1000, 5.0)
            pivot_strength[i] = strength
            
        if all(l[i] <= l[j] for j in range(i-left, i+right+1)):
            pl[i] = l[i]
            strength = min((min(l[i-left:i]) - l[i]) / l[i] * 1000, 5.0)
            pivot_strength[i] = strength
    
    return ph, pl, pivot_strength

def find_supply_demand_zones(df: pd.DataFrame, lookback: int = 100):
    """Find key supply and demand zones"""
    if len(df) < lookback:
        return {"supply": [], "demand": []}
    
    ph, pl, strength = detect_enhanced_pivots(df, 2, 2)
    
    supply_zones = []
    demand_zones = []
    
    # Find recent pivots
    recent_highs = [(i, ph[i], strength[i]) for i in range(len(ph)) if ph[i] is not None and i >= len(ph)-20]
    recent_lows = [(i, pl[i], strength[i]) for i in range(len(pl)) if pl[i] is not None and i >= len(pl)-20]
    
    # Cluster similar levels
    def cluster_levels(levels, tolerance=0.002):
        if not levels: return []
        clusters = []
        levels.sort(key=lambda x: x[1])
        
        current_cluster = [levels[0]]
        for level in levels[1:]:
            if abs(level[1] - current_cluster[0][1]) / current_cluster[0][1] <= tolerance:
                current_cluster.append(level)
            else:
                clusters.append(current_cluster)
                current_cluster = [level]
        clusters.append(current_cluster)
        return clusters
    
    high_clusters = cluster_levels(recent_highs)
    low_clusters = cluster_levels(recent_lows)
    
    for cluster in high_clusters:
        if len(cluster) >= 2:  # Need at least 2 touches
            prices = [item[1] for item in cluster]
            strengths = [item[2] for item in cluster]
            zone = {
                "price": sum(prices) / len(prices),
                "strength": sum(strengths) / len(strengths),
                "touches": len(cluster),
                "type": "supply"
            }
            supply_zones.append(zone)
    
    for cluster in low_clusters:
        if len(cluster) >= 2:
            prices = [item[1] for item in cluster]
            strengths = [item[2] for item in cluster]
            zone = {
                "price": sum(prices) / len(prices),
                "strength": sum(strengths) / len(strengths),
                "touches": len(cluster),
                "type": "demand"
            }
            demand_zones.append(zone)
    
    return {
        "supply": sorted(supply_zones, key=lambda x: x["strength"], reverse=True)[:3],
        "demand": sorted(demand_zones, key=lambda x: x["strength"], reverse=True)[:3]
    }

# ===== SMART COUNCIL SYSTEM =====
class SmartCouncil:
    def __init__(self):
        self.state = {"open": False, "side": None, "entry": None}
        self._last_log = None
        self._last_impulse = None
        self._last_pivot = None
        self._used_signals = set()  # Prevent signal repetition
        self._signal_history = deque(maxlen=20)
        self._vote_history = deque(maxlen=10)
        
    def _get_signal_hash(self, df, signal_type):
        """Create unique hash for signal to prevent repetition"""
        if len(df) < 2:
            return None
        current_bar = int(df["time"].iloc[-1])
        return f"{signal_type}_{current_bar}"
    
    def _can_use_signal(self, signal_hash):
        """Check if signal can be used (not recently used)"""
        if signal_hash in self._used_signals:
            return False
        self._used_signals.add(signal_hash)
        # Clean old signals (keep only last 50)
        if len(self._used_signals) > 50:
            self._used_signals = set(list(self._used_signals)[-50:])
        return True

    def enhanced_votes(self, df: pd.DataFrame, ind: dict, rf: dict):
        """Enhanced voting with MACD, VWAP, Delta Volume"""
        b = s = 0
        score = 0.0
        rb = []  # Reasons for buy
        rs = []  # Reasons for sell
        
        current_price = float(df["close"].iloc[-1])
        adx = float(ind.get("adx") or 0.0)
        rsi = float(ind.get("rsi") or 50.0)
        macd_hist = float(ind.get("macd_hist") or 0.0)
        vwap = float(ind.get("vwap") or 0.0)
        delta_volume = float(ind.get("delta_volume") or 0.0)
        volume_ratio = float(ind.get("volume_ratio") or 1.0)
        
        # 1. VWAP Trend Analysis
        if current_price > vwap and vwap > 0:
            b += VOTE_VWAP_TREND
            score += 0.8
            rb.append("Price>VWAP")
        elif current_price < vwap and vwap > 0:
            s += VOTE_VWAP_TREND
            score += 0.8
            rs.append("Price<VWAP")
        
        # 2. Delta Volume Analysis
        if delta_volume > 0:
            b += VOTE_DELTA_VOLUME
            score += 0.7
            rb.append(f"DeltaVol+{delta_volume:.0f}")
        elif delta_volume < 0:
            s += VOTE_DELTA_VOLUME
            score += 0.7
            rs.append(f"DeltaVol{delta_volume:.0f}")
        
        # 3. MACD Cross Analysis
        macd = float(ind.get("macd") or 0.0)
        macd_signal = float(ind.get("macd_signal") or 0.0)
        if macd > macd_signal and macd_hist > 0:
            b += VOTE_MACD_CROSS
            score += 0.9
            rb.append("MACD↑")
        elif macd < macd_signal and macd_hist < 0:
            s += VOTE_MACD_CROSS
            score += 0.9
            rs.append("MACD↓")
        
        # 4. Volume Surge
        if volume_ratio > IMPULSE_VOLUME_RATIO:
            if current_price > float(df["open"].iloc[-1]):
                b += VOTE_VOLUME_SURGE
                score += 0.6
                rb.append(f"VolSurge{volume_ratio:.1f}x")
            else:
                s += VOTE_VOLUME_SURGE
                score += 0.6
                rs.append(f"VolSurge{volume_ratio:.1f}x")
        
        # 5. Trend Alignment
        if adx > 25:
            if ind.get("plus_di", 0) > ind.get("minus_di", 0):
                b += VOTE_TREND_ALIGN
                score += 0.5
                rb.append("Trend↑")
            else:
                s += VOTE_TREND_ALIGN
                score += 0.5
                rs.append("Trend↓")
        
        # 6. RSI Momentum
        if 30 <= rsi <= 70:  # Avoid extremes
            if rsi > 55 and current_price > float(df["open"].iloc[-1]):
                b += 1
                score += 0.4
                rb.append("RSI_momentum↑")
            elif rsi < 45 and current_price < float(df["open"].iloc[-1]):
                s += 1
                score += 0.4
                rs.append("RSI_momentum↓")
        
        # 7. Existing RF and Structure Signals (keep original logic)
        boxes = find_supply_demand_zones(df)
        current_low = float(df["low"].iloc[-1])
        current_high = float(df["high"].iloc[-1])
        
        # Check demand zone bounce
        for zone in boxes.get("demand", []):
            if current_low <= zone["price"] * 1.002 and current_price > zone["price"]:
                b += VOTE_DEMAND_REJECT
                score += zone["strength"] * 0.5
                rb.append(f"DemandZone{zone['strength']:.1f}")
        
        # Check supply zone rejection
        for zone in boxes.get("supply", []):
            if current_high >= zone["price"] * 0.998 and current_price < zone["price"]:
                s += VOTE_SUPPLY_REJECT
                score += zone["strength"] * 0.5
                rs.append(f"SupplyZone{zone['strength']:.1f}")
        
        # RF Signals
        if rf.get("long"):
            signal_hash = self._get_signal_hash(df, "RF_LONG")
            if self._can_use_signal(signal_hash):
                b += VOTE_RF_CONFIRM
                score += rf.get("strength", 1.0) * 0.8
                rb.append(f"RF_LONG{rf.get('strength', 1.0):.1f}")
        
        if rf.get("short"):
            signal_hash = self._get_signal_hash(df, "RF_SHORT")
            if self._can_use_signal(signal_hash):
                s += VOTE_RF_CONFIRM
                score += rf.get("strength", 1.0) * 0.8
                rs.append(f"RF_SHORT{rf.get('strength', 1.0):.1f}")
        
        self._last_log = f"🏛️ SMART COUNCIL | BUY={b} [{', '.join(rb) or '—'}] | SELL={s} [{', '.join(rs) or '—'}] | Score={score:.2f} | ADX={adx:.1f}"
        print(colored(self._last_log, "green" if b > s else "red" if s > b else "cyan"))
        
        # Record vote history
        self._vote_history.append({
            'timestamp': time.time(),
            'buy_votes': b,
            'sell_votes': s,
            'score': score,
            'price': current_price
        })
        
        return b, s, score

    def decide_entry(self, df, ind, rf):
        """Enhanced entry decision with multiple confirmations"""
        b, s, score = self.enhanced_votes(df, ind, rf)
        adx = float(ind.get("adx") or 0.0)
        trend_strength = float(ind.get("trend_strength") or 0.0)
        
        entry = None
        if not self.state["open"]:
            # Stricter entry conditions
            min_votes = ENTRY_VOTES_MIN
            min_score = ENTRY_SCORE_MIN
            min_adx = ENTRY_ADX_MIN
            
            # Adjust requirements based on market conditions
            if trend_strength > 30:  # Strong trend
                min_votes -= 1  # Easier entry in strong trends
                min_score -= 1.0
            
            if b >= min_votes and score >= min_score and adx >= min_adx:
                self.state.update({
                    "open": True, 
                    "side": "long", 
                    "entry": float(df['close'].iloc[-1]),
                    "entry_time": time.time(),
                    "entry_votes": b,
                    "entry_score": score
                })
                entry = {"side": "buy", "reason": self._last_log}
                
            elif s >= min_votes and score >= min_score and adx >= min_adx:
                self.state.update({
                    "open": True, 
                    "side": "short", 
                    "entry": float(df['close'].iloc[-1]),
                    "entry_time": time.time(),
                    "entry_votes": s,
                    "entry_score": score
                })
                entry = {"side": "sell", "reason": self._last_log}
        
        return {"entry": entry, "exit": None, "log": self._last_log}

    def decide_exit(self, df, ind, rf, current_profit_pct):
        """Smart exit decision making"""
        if not self.state["open"]:
            return None
        
        b, s, score = self.enhanced_votes(df, ind, rf)
        adx = float(ind.get("adx") or 0.0)
        current_side = self.state["side"]
        
        # Exit if opposite votes are strong
        if current_side == "long" and s >= EXIT_VOTES_MIN and score >= 3.0:
            return {"exit": "sell", "reason": f"Opposite votes strong: {s} votes"}
        
        if current_side == "short" and b >= EXIT_VOTES_MIN and score >= 3.0:
            return {"exit": "buy", "reason": f"Opposite votes strong: {b} votes"}
        
        # Exit on trend weakness
        if adx < 18 and current_profit_pct > 0.5:  # Take profit when trend weakens
            return {"exit": "close", "reason": f"Trend weakening ADX={adx:.1f}"}
        
        # Exit on overextension
        rsi = float(ind.get("rsi") or 50.0)
        if current_side == "long" and rsi > 75 and current_profit_pct > 1.0:
            return {"exit": "close", "reason": f"Overbought RSI={rsi:.1f}"}
        
        if current_side == "short" and rsi < 25 and current_profit_pct > 1.0:
            return {"exit": "close", "reason": f"Oversold RSI={rsi:.1f}"}
        
        return None

# ===== ENHANCED POSITION MANAGEMENT =====
class SmartPositionManager:
    def __init__(self):
        self.breakeven_triggered = False
        self.trail_activated = False
        self.profit_targets = [0.4, 0.8, 1.5, 2.5]  # Dynamic targets
        self.targets_achieved = 0
        
    def calculate_dynamic_targets(self, atr_pct, trend_strength):
        """Calculate dynamic profit targets based on market conditions"""
        base_multipliers = [1.0, 2.0, 3.0, 4.0]
        if trend_strength > 30:
            base_multipliers = [x * 1.5 for x in base_multipliers]
        elif trend_strength < 15:
            base_multipliers = [x * 0.7 for x in base_multipliers]
            
        return [m * atr_pct for m in base_multipliers]
    
    def manage_take_profits(self, current_profit_pct, atr_pct, trend_strength):
        """Manage multiple take profit levels"""
        targets = self.calculate_dynamic_targets(atr_pct, trend_strength)
        close_fractions = [0.25, 0.35, 0.25, 0.15]  # Fractions to close at each target
        
        for i in range(self.targets_achieved, len(targets)):
            if current_profit_pct >= targets[i]:
                close_fraction = close_fractions[i]
                self.targets_achieved += 1
                return close_fraction, f"TP_{i+1}@{targets[i]:.2f}%"
        
        return 0.0, None
    
    def should_trail(self, current_profit_pct, highest_profit):
        """Determine if trailing should be activated"""
        if not self.trail_activated and current_profit_pct >= TRAIL_ACTIVATE_PCT:
            self.trail_activated = True
            return True
        return self.trail_activated
    
    def calculate_trail_level(self, price, side, atr, highest_profit_pct):
        """Calculate dynamic trail level"""
        if not self.trail_activated:
            return None
            
        trail_distance = atr * ATR_TRAIL_MULT
        if side == "long":
            return price - trail_distance
        else:
            return price + trail_distance

# ===== ENHANCED TRADING SYSTEM =====
smart_council = SmartCouncil()
position_manager = SmartPositionManager()

# Enhanced State
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0, "opp_votes": 0,
    "_last_entry_ts": 0, "_last_close_ts": 0, "_rf_debounce": 0,
    "_reversal_guard_bars": 0, "_last_flip_ts": 0,
    "entry_votes": 0, "entry_score": 0.0, "entry_time": 0
}

compound_pnl = 0.0
wait_for_next_signal_side = None
RESTART_HOLD_UNTIL_BAR = 0
_trades_timestamps = []

def _within_hour_rate_limit():
    now = time.time()
    while _trades_timestamps and now - _trades_timestamps[0] > 3600:
        _trades_timestamps.pop(0)
    return len(_trades_timestamps) < MAX_TRADES_PER_HOUR

def _mark_trade_timestamp():
    _trades_timestamps.append(time.time())

def compute_enhanced_size(balance, price, volatility):
    """Enhanced position sizing with volatility adjustment"""
    base_cap = (balance or 0.0) * RISK_ALLOC * LEVERAGE * SIZE_BUFFER
    
    # Adjust for volatility (reduce size in high volatility)
    vol_adjustment = max(0.5, min(1.5, 1.0 / (volatility * 10)))
    adjusted_cap = base_cap * vol_adjustment
    
    raw = max(0.0, adjusted_cap / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def open_enhanced_position(side, qty, price, tag=""):
    """Enhanced position opening with better risk management"""
    if qty <= 0:
        print(colored("❌ skip open (qty<=0)", "red"))
        return False
        
    if STATE["_reversal_guard_bars"] > 0 and side in ("buy", "sell"):
        print(colored("⛔ Reversal-Guard active — council-only entries", "yellow"))
        return False
        
    spr = orderbook_spread_bps()
    if spr is not None and (spr > HARD_SPREAD_BPS or spr > MAX_SPREAD_BPS):
        print(colored(f"⛔ spread {fmt(spr,2)}bps — guard", "yellow"))
        return False
        
    if not _within_hour_rate_limit():
        print(colored("⛔ rate-limit: too many trades/hour", "yellow"))
        return False
    
    _, _, mid = _best_quotes()
    
    if MODE_LIVE and USE_LIMIT_IOC:
        limit_price = _ioc_price(side, mid, MAX_SLIP_OPEN_BPS)
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            _create_order_ioc(SYMBOL, side, qty, limit_price, reduce_only=False)
        except Exception as e:
            print(colored(f"❌ IOC open fail: {e}", "red"))
            logging.error(e)
            return False
    elif MODE_LIVE:
        try:
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        except Exception as e:
            print(colored(f"❌ market open fail: {e}", "red"))
            logging.error(e)
            return False
            
    STATE.update({
        "open": True, 
        "side": "long" if side == "buy" else "short", 
        "entry": price,
        "qty": qty, 
        "pnl": 0.0, 
        "bars": 0, 
        "trail": None, 
        "breakeven": None,
        "tp1_done": False, 
        "highest_profit_pct": 0.0, 
        "profit_targets_achieved": 0,
        "opp_votes": 0, 
        "_last_entry_ts": int(time.time()),
        "entry_time": time.time()
    })
    
    _mark_trade_timestamp()
    print(colored(f"🚀 ENHANCED OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)} {tag}", "green" if side=='buy' else 'red'))
    
    if AUTOSAVE_ON_ORDER:
        save_state(tag="enhanced_open")
    return True

def smart_profit_taking(df, ind, current_profit_pct):
    """Enhanced profit taking based on multiple factors"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return
        
    price = float(df["close"].iloc[-1])
    atr_pct = (float(ind.get("atr") or 0.0) / price) * 100
    trend_strength = float(ind.get("trend_strength") or 0.0)
    
    # Check for dynamic profit taking
    close_fraction, reason = position_manager.manage_take_profits(
        current_profit_pct, atr_pct, trend_strength
    )
    
    if close_fraction > 0:
        close_partial(close_fraction, reason)
        
    # Check for exhaustion signals
    if detect_exhaustion(df, ind, STATE["side"], current_profit_pct):
        close_partial(0.5, "Exhaustion signal")
        
    # Update trailing stop
    if position_manager.should_trail(current_profit_pct, STATE["highest_profit_pct"]):
        trail_level = position_manager.calculate_trail_level(
            price, STATE["side"], float(ind.get("atr") or 0.0), STATE["highest_profit_pct"]
        )
        STATE["trail"] = trail_level

def detect_exhaustion(df, ind, side, profit_pct):
    """Detect exhaustion signals for smart profit taking"""
    if len(df) < 3:
        return False
        
    o, h, l, c = map(float, df[["open", "high", "low", "close"]].iloc[-1])
    rsi = float(ind.get("rsi") or 50.0)
    volume_ratio = float(ind.get("volume_ratio") or 1.0)
    
    # Large wick against position
    rng = max(h - l, 1e-12)
    if side == "long":
        upper_wick = h - max(o, c)
        if upper_wick / rng > 0.6 and profit_pct > 1.0:
            return True
    else:
        lower_wick = min(o, c) - l
        if lower_wick / rng > 0.6 and profit_pct > 1.0:
            return True
            
    # Volume divergence
    if volume_ratio > 2.0 and profit_pct > 2.0:
        return True
        
    # RSI extreme with profit
    if ((side == "long" and rsi > 75) or (side == "short" and rsi < 25)) and profit_pct > 1.5:
        return True
        
    return False

# ===== ENHANCED MAIN LOOP =====
app = Flask(__name__)

def enhanced_trade_loop():
    """Enhanced main trading loop with smart decision making"""
    global wait_for_next_signal_side, RESTART_HOLD_UNTIL_BAR, compound_pnl
    
    reconcile_state_with_exchange()
    last_decision_bar_time = 0
    
    while True:
        try:
            # Fetch market data
            bal = balance_usdt()
            df = fetch_ohlcv()
            ind = compute_enhanced_indicators(df)
            rf = rf_signal_enhanced(df)
            spread = orderbook_spread_bps()
            px = price_now() or rf["price"] or STATE.get("entry") or 0.0
            
            # Update PnL if position open
            if STATE["open"] and px:
                entry = STATE["entry"]
                if STATE["side"] == "long":
                    STATE["pnl"] = (px - entry) * STATE["qty"]
                else:
                    STATE["pnl"] = (entry - px) * STATE["qty"]
                    
                current_profit_pct = (STATE["pnl"] / (entry * STATE["qty"])) * 100
                if current_profit_pct > STATE["highest_profit_pct"]:
                    STATE["highest_profit_pct"] = current_profit_pct
            
            # Smart Council Decision
            council_decision = smart_council.decide_entry(df, ind, rf)
            council_log = council_decision.get("log")
            
            # Smart Exit Decisions
            if STATE["open"]:
                exit_decision = smart_council.decide_exit(df, ind, rf, current_profit_pct)
                if exit_decision:
                    print(colored(f"🎯 SMART EXIT: {exit_decision['reason']}", "yellow"))
                    close_market_strict(f"COUNCIL_EXIT: {exit_decision['reason']}")
            
            # Enhanced Profit Taking
            if STATE["open"]:
                smart_profit_taking(df, ind, current_profit_pct)
            
            # Manage position with enhanced logic
            manage_enhanced_position(df, ind, {"price": px, **rf})
            
            # Entry Logic
            decision_time = rf["time"]
            new_bar = decision_time != last_decision_bar_time
            
            if new_bar:
                last_decision_bar_time = decision_time
                if STATE["_reversal_guard_bars"] > 0:
                    STATE["_reversal_guard_bars"] -= 1
                if RESTART_HOLD_UNTIL_BAR > 0:
                    RESTART_HOLD_UNTIL_BAR -= 1
                
                # Enhanced Entry Conditions
                if not STATE["open"] and RESTART_HOLD_UNTIL_BAR <= 0:
                    sig = None
                    tag = ""
                    
                    if council_decision["entry"]:
                        sig = council_decision["entry"]["side"]
                        tag = f"[SMART_COUNCIL] {council_decision['entry']['reason']}"
                    
                    if sig:
                        if wait_for_next_signal_side and sig != wait_for_next_signal_side:
                            print(colored(f"⏳ Waiting for {wait_for_next_signal_side.upper()} signal", "cyan"))
                        elif (time.time() - STATE.get("_last_close_ts", 0)) < CLOSE_COOLDOWN_S:
                            cooldown_left = CLOSE_COOLDOWN_S - (time.time() - STATE.get("_last_close_ts", 0))
                            print(colored(f"⏳ Cooldown: {int(cooldown_left)}s remaining", "cyan"))
                        elif not _within_hour_rate_limit():
                            print(colored("⏳ Rate limit: too many trades this hour", "cyan"))
                        else:
                            # Enhanced position sizing
                            atr_volatility = (float(ind.get("atr") or 0.0) / (px or 1.0))
                            qty = compute_enhanced_size(bal, px or rf["price"], atr_volatility)
                            
                            if qty > 0 and (px or rf["price"]):
                                if open_enhanced_position(sig, qty, px or rf["price"], tag):
                                    wait_for_next_signal_side = None
                                    position_manager.__init__()  # Reset for new position
                            else:
                                print(colored("⚠️ Invalid quantity or price for entry", "yellow"))
            
            # Enhanced UI Display
            display_enhanced_dashboard(bal, {"price": px, **rf}, ind, spread, df, council_log)
            
            # Update bars count
            if len(df) >= 2 and int(df["time"].iloc[-1]) != int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"] += 1
            
            # Auto-save
            if AUTOSAVE_EVERY_LOOP:
                save_state(tag="enhanced_loop")
                
            # Adaptive sleep
            sleep_time = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_time)
            
        except Exception as e:
            print(colored(f"❌ Enhanced loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"Enhanced loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

def display_enhanced_dashboard(bal, info, ind, spread_bps, df=None, council_log=None):
    """Enhanced dashboard display"""
    left_s = time_to_candle_close(df) if df is not None else 0
    
    print(colored("═" * 120, "cyan"))
    print(colored(f"🎯 SMART COUNCIL PRO — {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", "cyan"))
    print(colored("═" * 120, "cyan"))
    
    # Market Data Section
    print("📊 ENHANCED MARKET DATA")
    print(f"   💲 Price {fmt(info.get('price'))} | VWAP={fmt(ind.get('vwap'))} | Spread={fmt(spread_bps,2)}bps")
    print(f"   📈 RSI={fmt(ind.get('rsi'))} | ADX={fmt(ind.get('adx'))} | ATR={fmt(ind.get('atr'))} | VEI={fmt(ind.get('vei'),2)}")
    print(f"   🔄 MACD={fmt(ind.get('macd'),4)} | Signal={fmt(ind.get('macd_signal'),4)} | Hist={fmt(ind.get('macd_hist'),4)}")
    print(f"   📊 DeltaVol={fmt(ind.get('delta_volume'),0)} | VolRatio={fmt(ind.get('volume_ratio'),2)} | TrendStr={fmt(ind.get('trend_strength'),2)}")
    
    if council_log:
        print(colored(f"   🏛️  {council_log}", "white"))
    
    # Position Section
    print("\n💼 ENHANCED POSITION")
    equity = (bal or 0) + compound_pnl
    bal_line = f"Balance={fmt(bal,2)} | Equity={fmt(equity,2)} | CompoundPnL={fmt(compound_pnl)} | Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x"
    print(colored(f"   {bal_line}", "yellow"))
    
    if STATE["open"]:
        side_icon = '🟩 LONG' if STATE['side'] == 'long' else '🟥 SHORT'
        profit_pct = ((STATE['pnl'] / (STATE['entry'] * STATE['qty'])) * 100) if STATE['entry'] and STATE['qty'] else 0
        
        print(f"   {side_icon} Entry={fmt(STATE['entry'])} | Qty={fmt(STATE['qty'],4)} | PnL={fmt(STATE['pnl'])} ({fmt(profit_pct,2)}%)")
        print(f"   🎯 Bars={STATE['bars']} | Trail={fmt(STATE['trail'])} | BE={fmt(STATE['breakeven'])}")
        print(f"   🚀 TP_Levels={position_manager.targets_achieved}/{len(position_manager.profit_targets)} | HP={fmt(STATE['highest_profit_pct'],2)}%")
    else:
        print("   ⚪ FLAT")
        if wait_for_next_signal_side:
            print(colored(f"   ⏳ Waiting opposite signal: {wait_for_next_signal_side.upper()}", "cyan"))
    
    print(f"   ⏱️  Next candle in: {left_s}s")
    print(colored("═" * 120, "cyan"))

# ===== ENHANCED HTTP ENDPOINTS =====
@app.route("/")
def home():
    mode = 'LIVE' if MODE_LIVE else 'PAPER'
    return f"🎯 SMART COUNCIL PRO — {SYMBOL} {INTERVAL} — {mode} — AI Trading System"

@app.route("/metrics")
def enhanced_metrics():
    return jsonify({
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE,
        "risk_alloc": RISK_ALLOC,
        "price": price_now(),
        "state": STATE,
        "compound_pnl": compound_pnl,
        "council_log": smart_council._last_log,
        "position_manager": {
            "targets_achieved": position_manager.targets_achieved,
            "trail_activated": position_manager.trail_activated,
            "breakeven_triggered": position_manager.breakeven_triggered
        },
        "market_conditions": {
            "max_spread_bps": MAX_SPREAD_BPS,
            "hard_spread_bps": HARD_SPREAD_BPS,
            "pause_adx": PAUSE_ADX_THRESHOLD
        }
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "ts": datetime.utcnow().isoformat(),
        "open": STATE["open"],
        "side": STATE["side"],
        "qty": STATE["qty"],
        "system": "enhanced_smart_council"
    }), 200

# ===== ENHANCED KEEPALIVE =====
def enhanced_keepalive_loop():
    if not SELF_URL:
        print(colored("⛔ Enhanced keepalive disabled (no SELF_URL)", "yellow"))
        return
        
    import requests
    sess = requests.Session()
    sess.headers.update({"User-Agent": "smart-council-pro/keepalive"})
    
    print(colored(f"🔗 Enhanced keepalive every 50s → {SELF_URL}", "cyan"))
    
    while True:
        try:
            resp = sess.get(SELF_URL, timeout=8)
            if resp.status_code == 200:
                print(colored("✅ Keepalive successful", "green"))
            else:
                print(colored(f"⚠️ Keepalive status: {resp.status_code}", "yellow"))
        except Exception as e:
            print(colored(f"⚠️ Keepalive error: {e}", "yellow"))
            
        time.sleep(50)

# ===== ENHANCED BOOT =====
if __name__ == "__main__":
    print(colored("🎯 SMART COUNCIL PRO TRADING BOT", "green"))
    print(colored("═" * 80, "green"))
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • {SYMBOL} • {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}%×{LEVERAGE}x • ENTRY: Smart Council + Enhanced RF", "yellow"))
    print(colored("FEATURES: MACD + VWAP + Delta Volume + Smart Profit Taking + AI Decisions", "yellow"))
    print(colored("═" * 80, "green"))
    
    logging.info("Smart Council Pro starting...")
    
    # Signal handling
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    
    # Start enhanced threads
    import threading
    threading.Thread(target=enhanced_trade_loop, daemon=True).start()
    threading.Thread(target=enhanced_keepalive_loop, daemon=True).start()
    
    # Start Flask app
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
