# file: sui_bot_council_elite_pro_enhanced.py
# -*- coding: utf-8 -*-
"""
BYBIT — SUI Perp Council ELITE PRO PLUS (المتداول المحترف المتكامل)
- نظام متكامل لإدارة الصفقات من الدخول إلى جني الأرباح
- تحليل متقدم للشموع ومناطق السيولة
- إدارة ذكية للربح والخسارة
- كشف القمم والقيعان الحقيقية
"""

import os, time, math, random, signal, sys, traceback, logging, uuid, threading, csv
from logging.handlers import RotatingFileHandler
from datetime import datetime
from collections import deque
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd
import ccxt
from flask import Flask, jsonify

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV ===================
API_KEY  = os.getenv("BYBIT_API_KEY", "")
API_SEC  = os.getenv("BYBIT_API_SECRET", "")
SELF_URL = (os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")).strip()
PORT     = int(os.getenv("PORT", "5000"))
MODE_LIVE = bool(API_KEY and API_SEC)

# =================== SETTINGS المحسنة ===================
SYMBOL        = "SUI/USDT:USDT"
INTERVAL      = "15m"

LEVERAGE      = 10
RISK_ALLOC    = 0.60
POSITION_MODE = "oneway"

# نظام إدارة الصفقات المحسن
TRADE_MANAGEMENT = {
    "partial_take_profit": True,  # جني أرباح جزئي
    "multi_targets": True,        # أهداف متعددة
    "dynamic_trailing": True,     # وقف خسارة متحرك ديناميكي
    "break_even": True,           # الانتقال إلى نقطة التعادل
}

# مستويات جني الأرباح
TAKE_PROFIT_LEVELS = [
    {"target": 0.8, "percentage": 0.30},   # الهدف 1: 0.8% ربح، جني 30% من المركز
    {"target": 1.5, "percentage": 0.40},   # الهدف 2: 1.5% ربح، جني 40% من المركز  
    {"target": 2.5, "percentage": 0.30},   # الهدف 3: 2.5% ربح، جني 30% من المركز
]

BREAK_EVEN_AT = 0.6  # الانتقال لنقطة التعادل عند 0.6% ربح
TRAIL_START_AT = 1.0  # بدء الوقف المتحرك عند 1.0% ربح

# RF
RF_SOURCE   = "close"
RF_PERIOD   = 20
RF_MULT     = 3.5
RF_HYST_BPS = 6.0

# المؤشرات الجديدة
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VWAP_WINDOW = 20

# الحماية
ADX_ENTRY_MIN   = 17.0
MAX_SPREAD_BPS  = 8.0
SPREAD_HARD_BPS = 15.0
ENTRY_GUARD_WINDOW_SEC = 6
CLOSE_GUARD_WINDOW_SEC = 3
COOLDOWN_SEC    = 90
REENTRY_COOLDOWN_SEC = 45
MAX_TRADES_PER_HOUR = 6

# منع التكرار
LAST_SIGNAL_USED = {
    "side": None,
    "bar_ts": None,
    "src": None,
    "strength": 0.0
}

# الترند والكسر
BREAK_HYST_BPS     = 10.0
BREAK_ADX_MIN      = 22.0
BREAK_DI_MARGIN    = 5.0
BREAK_BODY_ATR_MIN = 0.60
TREND_STRONG_ADX   = 28.0
TREND_STRONG_DI_M  = 8.0
OPP_RF_DEBOUNCE    = 2

# إدارة الصفقة المحسنة
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6

# جني الأرباح الذكي المحسن
WICK_TAKE_MIN_PCT   = 0.40
WICK_BIG_RATIO      = 0.62
BODY_BIG_ATR_MULT   = 1.10

# خروج الذكاء المحسن
EXH_MIN_PNL_PCT   = 0.35
EXH_ADX_DROP      = 6.0
EXH_ADX_MIN       = 18.0
EXH_RSI_PULLBACK  = 7.0
EXH_WICK_RATIO    = 0.60
EXH_HYST_MIN_BPS  = 8.0
EXH_BOS_LOOKBACK  = 6
EXH_VOTES_NEEDED  = 3

# =================== إعدادات مجلس الإدارة المحسنة ===================
COUNCIL_ENTRY_VOTES_MIN  = 7  # زيادة من 6 إلى 7
COUNCIL_STRONG_SCORE_MIN = 4.5  # زيادة من 4.0 إلى 4.5

# شروط إضافية للقوة
MIN_CONFIRMATION_SIGNALS = 4  # عدد الإشارات المؤكدة المطلوبة
TREND_ALIGNMENT_BONUS = 1.5   # مكافأة محاذاة الترند
VOLUME_CONFIRMATION_REQUIRED = True  # تأكيد الحجم مطلوب

# الانزلاق
MAX_SLIP_OPEN_BPS   = 25.0
MAX_SLIP_CLOSE_BPS  = 35.0

# VEI الانفجار
VEI_LEN_BASE      = 50
VEI_EXPLODE_MULT  = 2.2
VEI_FILTER_BPS    = 12.0
VEI_ADX_MIN       = 18.0
VEI_VOL_VOTE      = 1

# التوقيت
BASE_SLEEP   = 3
NEAR_CLOSE_S = 1
MIN_SIGNAL_AGE_SEC = 1

# كشف التذبذب
BB_LEN                 = 20
CHOP_ADX_MAX           = 16.0
CHOP_LOOKBACK          = 120
CHOP_ATR_PCT_FRACTION  = 0.65
CHOP_BB_WIDTH_PCT_MAX  = 1.10
CHOP_RANGE_BARS        = 24
CHOP_RANGE_BPS_MAX     = 60.0
CHOP_MIN_PNL_PCT       = 0.20
CHOP_STRICT_MODE       = True
CHOP_STRONG_BREAK_BONUS= 2
POST_CHOP_WAIT_BARS    = 2
POST_CHOP_REQUIRE_RF   = True
MIN_REENTRY_BARS       = 1

# القمم والقيعان الحقيقية المحسنة
TTB_SWING_LEFT  = 2
TTB_SWING_RIGHT = 2
TTB_ADX_MIN     = 17.0
TTB_WICK_RAT    = 0.55
TTB_BODY_ATR    = 0.60
TTB_SCORE_MIN   = 3.5  # زيادة من 3.2 إلى 3.5

# Bookmap-lite
OBI_DEPTH = 10
OBI_ABS_MIN = 0.15
CVD_SMOOTH = 10

# المؤشرات الجديدة
MACD_TREND_THRESHOLD = 0.0
VWAP_TREND_WINDOW = 20
DELTA_VOLUME_SMOOTH = 14

# مناطق السيولة
LIQ_EQ_LOOKBACK   = 20
LIQ_EQ_TOL_BPS    = 8.0
SWEEP_WICK_RATIO  = 0.55
RETEST_MAX_BARS   = 8

# التسجيل
DECISIONS_CSV = Path("decisions_log.csv")

# =================== تسجيل الملفات ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready", "cyan"))
setup_file_logging()

# =================== المنصة ===================
def make_ex():
    return ccxt.bybit({
        "apiKey": API_KEY, "secret": API_SEC,
        "enableRateLimit": True, "timeout": 20000,
        "options": {"defaultType": "swap"}
    })
ex = make_ex()
MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN  = None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision", {}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits", {}) or {}).get("amount", {}).get("step", None)
        LOT_MIN  = (MARKET.get("limits", {}) or {}).get("amount", {}).get("min",  None)
        print(colored(f"🔧 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}", "yellow"))

def ensure_leverage_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            print(colored(f"✅ leverage set: {LEVERAGE}x", "green"))
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}", "yellow"))
        print(colored(f"📌 position mode: {POSITION_MODE}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_mode: {e}", "yellow"))

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}", "yellow"))

# =================== الأقفال والحماية ===================
ENTRY_LOCK = threading.Lock()
CLOSE_LOCK = threading.Lock()
ENTRY_IN_PROGRESS = False
CLOSE_IN_PROGRESS = False
PENDING_OPEN = False
_last_entry_attempt_ts = 0.0
_last_close_attempt_ts = 0.0
LAST_DECISION_BAR_TS = 0

LAST_CLOSE_TS = 0
TRADE_TIMES = deque(maxlen=10)
compound_pnl = 0.0

POST_CHOP_BLOCK_ACTIVE = False
POST_CHOP_BLOCK_UNTIL_BAR = 0
LAST_CLOSE_BAR_TS = 0

# =================== STATE المحسن مع إدارة الصفقات ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None,
    "hp_pct": 0.0, "strength": 0.0,
    "peak_adx": 0.0, "rsi_peak": 50.0, "rsi_trough": 50.0,
    "peak_price": 0.0, "trough_price": 0.0,
    "opp_rf_count": 0, "scm_line": "", "chop_flag": False,
    "cvd": 0.0, "plan": "SIT_OUT", "plan_reasons": [],
    "macd_trend": "neutral", "vwap_trend": "neutral", "delta_pressure": 0.0,
    
    # إدارة الصفقات المحسنة
    "trade_management": {
        "partial_taken": False,           # هل تم جني أرباح جزئي
        "targets_hit": [],                # الأهداف التي تم تحقيقها
        "break_even_moved": False,        # هل تم الانتقال لنقطة التعادل
        "trailing_active": False,         # هل الوقف المتحرك نشط
        "initial_stop": None,             # وقف الخسارة الأولي
        "current_stop": None,             # وقف الخسارة الحالي
    },
    "position_size": 0.0,                 # حجم المركز الإجمالي
    "remaining_size": 0.0,                # حجم المركز المتبقي بعد الجني الجزئي
}

def _now(): return time.time()
def _order_link(prefix="ORD"): return f"{prefix}-{uuid.uuid4().hex[:18]}"
def _norm_sym(s: str) -> str: return (s or "").replace("/", "").replace(":", "").upper()
def _sym_match(a: str, b: str) -> bool:
    A, B = _norm_sym(a), _norm_sym(b); return A == B or A in B or B in A

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = AMT_PREC
        if (not prec or prec<=0) and LOT_MIN and LOT_MIN < 1:
            try: prec = max(1, -Decimal(str(LOT_MIN)).as_tuple().exponent)
            except Exception: prec = 1
        d = d.quantize(Decimal(1).scaleb(-int(prec or 0)), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except Exception:
        return max(0.0, float(q))

def safe_qty(q):
    q = _round_amt(q)
    if q<=0: print(colored(f"⚠️ qty invalid after normalize → {q}", "yellow"))
    return q

def with_retry(fn, tries=3, base_wait=0.35):
    for i in range(tries):
        try: return fn()
        except Exception:
            if i==tries-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.2)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob.get("bids") else None
        ask = ob["asks"][0][0] if ob.get("asks") else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0 if mid else None
    except Exception:
        return None

def _interval_seconds(iv: str) -> int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df) == 0: return tf
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    while next_close_ms <= now_ms: next_close_ms += tf*1000
    return int(max(0, next_close_ms - now_ms)/1000)

def _best_bid_ask():
    ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=OBI_DEPTH))
    bid = ob["bids"][0][0] if ob["bids"] else None
    ask = ob["asks"][0][0] if ob["asks"] else None
    return bid, ask, ob

def _price_band(side:str, px:float, max_bps:float):
    if px is None: return None
    if side == "buy":  return px * (1 + max_bps/10000.0)
    else:              return px * (1 - max_bps/10000.0)

# =================== المؤشرات المحسنة ===================
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()

def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

def compute_vwap(df: pd.DataFrame, window=20):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
    return vwap

def compute_delta_volume(df: pd.DataFrame, smooth=14):
    delta = df['volume'] * ((df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1))
    delta_smooth = delta.rolling(smooth).mean()
    return delta, delta_smooth

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN, MACD_SLOW) + 3:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0,
                "macd_line":0.0,"macd_signal":0.0,"macd_hist":0.0,"vwap":0.0,"delta_vol":0.0}
    
    c,h,l,v = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float), df["volume"].astype(float)
    
    # ATR
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)
    
    # RSI
    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))
    
    # ADX
    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wilder_ema(plus_dm, ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wilder_ema(minus_dm, ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wilder_ema(dx, ADX_LEN)
    
    # MACD
    macd_line, macd_signal, macd_hist = compute_macd(c, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    
    # VWAP
    vwap = compute_vwap(df, VWAP_WINDOW)
    
    # Delta Volume
    delta_vol, delta_vol_smooth = compute_delta_volume(df, DELTA_VOLUME_SMOOTH)
    
    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i]),
        "macd_line": float(macd_line.iloc[i]), "macd_signal": float(macd_signal.iloc[i]), 
        "macd_hist": float(macd_hist.iloc[i]), "vwap": float(vwap.iloc[i]),
        "delta_vol": float(delta_vol_smooth.iloc[i])
    }

# =================== نظام إدارة الصفقات المحسن ===================
def calculate_position_size(balance, price, strength):
    """حساب حجم المركز بناء على قوة الإشارة ورأس المال"""
    base_size = compute_size(balance, price)
    
    # تعديل الحجم بناء على قوة الإشارة
    strength_factor = min(1.0, max(0.5, strength / 5.0))
    adjusted_size = base_size * strength_factor
    
    return safe_qty(adjusted_size)

def setup_trade_management(entry_price, atr, side, strength):
    """إعداد نظام إدارة الصفقة"""
    # وقف الخسارة الأولي (2x ATR)
    stop_distance = atr * 2.0
    if side == "long":
        initial_stop = entry_price - stop_distance
    else:
        initial_stop = entry_price + stop_distance
    
    STATE["trade_management"].update({
        "partial_taken": False,
        "targets_hit": [],
        "break_even_moved": False,
        "trailing_active": False,
        "initial_stop": initial_stop,
        "current_stop": initial_stop,
    })
    
    print(colored(f"🎯 إدارة الصفقة: وقف أولي {fmt(initial_stop)} | ATR {fmt(atr)}", "cyan"))

def check_take_profit_targets(current_price, entry_price, side, atr):
    """التحقق من مستويات جني الأرباح"""
    if not TRADE_MANAGEMENT["partial_take_profit"]:
        return False
    
    # حساب الربح الحالي
    if side == "long":
        profit_pct = (current_price - entry_price) / entry_price * 100
    else:
        profit_pct = (entry_price - current_price) / entry_price * 100
    
    tm = STATE["trade_management"]
    remaining_qty = STATE["remaining_size"] or STATE["qty"]
    
    for level in TAKE_PROFIT_LEVELS:
        target = level["target"]
        percentage = level["percentage"]
        
        if target not in tm["targets_hit"] and profit_pct >= target:
            # جني نسبة من المركز
            close_qty = safe_qty(remaining_qty * percentage)
            if close_qty > 0:
                close_partial_position(close_qty, f"TAKE_PROFIT_{target}%")
                tm["targets_hit"].append(target)
                print(colored(f"🎯 جني ربح {target}%: إغلاق {percentage*100}% من المركز", "green"))
                return True
    
    return False

def check_break_even(current_price, entry_price, side, atr):
    """التحقق من الانتقال إلى نقطة التعادل"""
    if not TRADE_MANAGEMENT["break_even"]:
        return False
    
    tm = STATE["trade_management"]
    if tm["break_even_moved"]:
        return False
    
    # حساب الربح الحالي
    if side == "long":
        profit_pct = (current_price - entry_price) / entry_price * 100
        new_stop = entry_price
    else:
        profit_pct = (entry_price - current_price) / entry_price * 100
        new_stop = entry_price
    
    if profit_pct >= BREAK_EVEN_AT:
        tm["current_stop"] = new_stop
        tm["break_even_moved"] = True
        print(colored(f"🛡️ الانتقال لنقطة التعادل: وقف الخسارة {fmt(new_stop)}", "yellow"))
        return True
    
    return False

def update_trailing_stop(current_price, entry_price, side, atr):
    """تحديث الوقف المتحرك"""
    if not TRADE_MANAGEMENT["dynamic_trailing"]:
        return
    
    tm = STATE["trade_management"]
    
    # حساب الربح الحالي
    if side == "long":
        profit_pct = (current_price - entry_price) / entry_price * 100
        if profit_pct >= TRAIL_START_AT:
            new_stop = current_price - (atr * ATR_TRAIL_MULT)
            if new_stop > tm["current_stop"]:
                tm["current_stop"] = new_stop
                tm["trailing_active"] = True
                print(colored(f"📈 تحديث الوقف المتحرك: {fmt(new_stop)}", "blue"))
    else:
        profit_pct = (entry_price - current_price) / entry_price * 100
        if profit_pct >= TRAIL_START_AT:
            new_stop = current_price + (atr * ATR_TRAIL_MULT)
            if new_stop < tm["current_stop"]:
                tm["current_stop"] = new_stop
                tm["trailing_active"] = True
                print(colored(f"📈 تحديث الوقف المتحرك: {fmt(new_stop)}", "blue"))

def check_stop_loss(current_price, side):
    """التحقق من وقف الخسارة"""
    tm = STATE["trade_management"]
    stop_price = tm["current_stop"]
    
    if side == "long" and current_price <= stop_price:
        close_market_strict(f"STOP_LOSS {fmt(stop_price)}")
        return True
    elif side == "short" and current_price >= stop_price:
        close_market_strict(f"STOP_LOSS {fmt(stop_price)}")
        return True
    
    return False

def close_partial_position(qty, reason):
    """إغلاق جزئي للمركز"""
    global CLOSE_IN_PROGRESS, _last_close_attempt_ts
    
    if CLOSE_LOCK.locked() or CLOSE_IN_PROGRESS:
        print(colored("⏸️ close in progress — skip partial", "yellow"))
        return False
    
    if _now() - _last_close_attempt_ts < CLOSE_GUARD_WINDOW_SEC:
        print(colored("⏸️ close guard window — skip partial", "yellow"))
        return False
    
    with CLOSE_LOCK:
        CLOSE_IN_PROGRESS = True
        _last_close_attempt_ts = _now()
        
        try:
            side_to_close = "sell" if STATE["side"] == "long" else "buy"
            qty_to_close = safe_qty(qty)
            
            if qty_to_close <= 0:
                print(colored("⚠️ partial close qty too small", "yellow"))
                return False
            
            # تحديث حجم المركز المتبقي
            STATE["remaining_size"] = safe_qty(STATE["qty"] - qty_to_close)
            STATE["trade_management"]["partial_taken"] = True
            
            link = _order_link("PART")
            if MODE_LIVE:
                params = _params_close()
                params.update({"orderLinkId": link})
                ex.create_order(SYMBOL, "market", side_to_close, qty_to_close, None, params)
            else:
                print(colored(f"[PAPER] partial close {side_to_close} {qty_to_close}", "cyan"))
            
            time.sleep(0.5)
            print(colored(f"✅ إغلاق جزئي: {fmt(qty_to_close,4)} | السبب: {reason}", "green"))
            return True
            
        except Exception as e:
            print(colored(f"❌ partial close error: {e}", "red"))
            return False
        finally:
            CLOSE_IN_PROGRESS = False

# =================== نظام الشموع المحسن ===================
def analyze_candle_strength(df: pd.DataFrame, ind: dict) -> Dict[str, float]:
    """تحليل قوة الشمعة الحالية"""
    if len(df) < 3:
        return {"strength": 0.0, "momentum": 0.0, "volume_power": 0.0}
    
    d = df.iloc[:-1]
    o=float(d["open"].iloc[-1]); c=float(d["close"].iloc[-1])
    h=float(d["high"].iloc[-1]); l=float(d["low"].iloc[-1])
    v=float(d["volume"].iloc[-1])
    
    # متوسط الحجم لآخر 20 شمعة
    avg_volume = df['volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else v
    
    # حساب قوة الشمعة
    body_size = abs(c - o)
    total_range = h - l
    body_ratio = body_size / total_range if total_range > 0 else 0
    
    # قوة الزخم
    momentum = 0.0
    if c > o:  # شمعة صاعدة
        momentum = (c - o) / o * 100
    else:  # شمعة هابطة
        momentum = (o - c) / o * 100
    
    # قوة الحجم
    volume_power = v / avg_volume if avg_volume > 0 else 1.0
    
    # النتيجة النهائية
    strength = (body_ratio * 0.4 + min(abs(momentum) * 2, 1.0) * 0.4 + min(volume_power, 2.0) * 0.2)
    
    return {
        "strength": strength,
        "momentum": momentum,
        "volume_power": volume_power,
        "body_ratio": body_ratio
    }

def detect_strong_candle_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """كشف أنماط الشموع القوية"""
    patterns = {
        "strong_bullish": False,
        "strong_bearish": False,
        "hammer": False,
        "shooting_star": False,
        "engulfing_bull": False,
        "engulfing_bear": False
    }
    
    if len(df) < 3:
        return patterns
    
    d = df.iloc[:-1]
    o1, c1, h1, l1 = float(d["open"].iloc[-1]), float(d["close"].iloc[-1]), float(d["high"].iloc[-1]), float(d["low"].iloc[-1])
    o0, c0, h0, l0 = float(d["open"].iloc[-2]), float(d["close"].iloc[-2]), float(d["high"].iloc[-2]), float(d["low"].iloc[-2])
    
    # شمعة صاعدة قوية
    body1 = abs(c1 - o1)
    total_range1 = h1 - l1
    if c1 > o1 and body1 / total_range1 > 0.7 and c1 > h0:
        patterns["strong_bullish"] = True
    
    # شمعة هابطة قوية
    if c1 < o1 and body1 / total_range1 > 0.7 and c1 < l0:
        patterns["strong_bearish"] = True
    
    # نمط المطرقة
    lower_shadow = min(o1, c1) - l1
    upper_shadow = h1 - max(o1, c1)
    if lower_shadow > 2 * body1 and upper_shadow < body1 * 0.3 and c1 > o1:
        patterns["hammer"] = True
    
    # نمط النجمه الساقطه
    if upper_shadow > 2 * body1 and lower_shadow < body1 * 0.3 and c1 < o1:
        patterns["shooting_star"] = True
    
    # نمط الابتلاع الصاعد
    if c1 > o1 and o1 < c0 and c1 > o0 and c0 < o0:
        patterns["engulfing_bull"] = True
    
    # نمط الابتلاع الهابط
    if c1 < o1 and o1 > c0 and c1 < o0 and c0 > o0:
        patterns["engulfing_bear"] = True
    
    return patterns

# =================== نظام الدخول المحسن ===================
def enhanced_entry_decision(candidates, df, ind, info):
    """اتخاذ قرار دخول محسن مع تحليل متقدم"""
    if not candidates:
        return None
    
    # تحليل قوة الشموع
    candle_strength = analyze_candle_strength(df, ind)
    candle_patterns = detect_strong_candle_patterns(df)
    
    # تحليل الحجم
    volume_analysis = analyze_volume(df)
    
    # ترشيح المرشحين بناء على قوة إضافية
    strong_candidates = []
    
    for candidate in candidates:
        strength_score = candidate.get("score", 0)
        votes = candidate.get("votes", 0)
        
        # عوامل التعزيز
        boost_factors = 0
        
        # تعزيز بناء على قوة الشمعة
        if candle_strength["strength"] > 0.7:
            boost_factors += 1
        
        # تعزيز بناء على أنماط الشموع
        if (candidate["side"] == "buy" and candle_patterns["strong_bullish"]) or \
           (candidate["side"] == "sell" and candle_patterns["strong_bearish"]):
            boost_factors += 2
        
        if (candidate["side"] == "buy" and candle_patterns["engulfing_bull"]) or \
           (candidate["side"] == "sell" and candle_patterns["engulfing_bear"]):
            boost_factors += 1
        
        # تعزيز بناء على الحجم
        if volume_analysis["volume_ok"]:
            boost_factors += 1
        
        # تعزيز بناء على محاذاة الترند
        trend = trend_context(ind)
        if (candidate["side"] == "buy" and trend in ["up", "strong_up"]) or \
           (candidate["side"] == "sell" and trend in ["down", "strong_down"]):
            boost_factors += 1
        
        # تعزيز بناء على المؤشرات
        if abs(float(ind.get("macd_hist", 0))) > 0.002:
            boost_factors += 1
        
        # حساب القوة النهائية
        final_strength = strength_score + (boost_factors * 0.3)
        
        # إضافة مرشح مع القوة النهائية
        enhanced_candidate = candidate.copy()
        enhanced_candidate["final_strength"] = final_strength
        enhanced_candidate["boost_factors"] = boost_factors
        
        strong_candidates.append(enhanced_candidate)
    
    # ترتيب المرشحين حسب القوة النهائية
    strong_candidates.sort(key=lambda x: x["final_strength"], reverse=True)
    
    # اختيار أفضل مرشح
    if strong_candidates and strong_candidates[0]["final_strength"] >= 4.5:
        return strong_candidates[0]
    
    return None

def analyze_volume(df: pd.DataFrame) -> Dict[str, any]:
    """تحليل الحجم المتقدم"""
    if len(df) < 21:
        return {"volume_ok": False, "volume_ratio": 1.0, "volume_trend": "neutral"}
    
    d = df.iloc[:-1]
    current_volume = float(d["volume"].iloc[-1])
    avg_volume_20 = float(df["volume"].rolling(20).mean().iloc[-2])
    
    volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
    
    # اتجاه الحجم
    volume_trend = "neutral"
    if volume_ratio > 1.5:
        volume_trend = "strong"
    elif volume_ratio > 1.2:
        volume_trend = "rising"
    elif volume_ratio < 0.8:
        volume_trend = "falling"
    
    return {
        "volume_ok": volume_ratio > 1.2,
        "volume_ratio": volume_ratio,
        "volume_trend": volume_trend
    }

# =================== إدارة الصفقة المحسنة ===================
def enhanced_manage_position(df, ind, info, zones, trend):
    """إدارة محسنة للصفقة مع نظام جني الأرباح المتعدد"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return
    
    current_price = info["price"]
    entry_price = STATE["entry"]
    side = STATE["side"]
    atr = float(ind.get("atr", 0))
    
    # 1. التحقق من وقف الخسارة
    if check_stop_loss(current_price, side):
        return
    
    # 2. التحقق من جني الأرباح الجزئي
    if check_take_profit_targets(current_price, entry_price, side, atr):
        return
    
    # 3. التحقق من الانتقال لنقطة التعادل
    if check_break_even(current_price, entry_price, side, atr):
        return
    
    # 4. تحديث الوقف المتحرك
    update_trailing_stop(current_price, entry_price, side, atr)
    
    # 5. الإدارة التقليدية (من النسخة الأصلية)
    traditional_manage_position(df, ind, info, zones, trend)

def traditional_manage_position(df, ind, info, zones, trend):
    """الإدارة التقليدية للصفقة (من النسخة الأصلية)"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return
    
    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    rr = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    
    opp = (side == "long" and info.get("short")) or (side == "short" and info.get("long"))
    STATE["opp_rf_count"] = STATE.get("opp_rf_count", 0) + 1 if opp else 0
    
    if wick_or_bigcandle_harvest(df, ind, info):
        return
    
    choppy = is_chop_zone(df, ind)
    STATE["chop_flag"] = bool(choppy)
    
    if choppy and rr >= CHOP_MIN_PNL_PCT and trend not in ("strong_up", "strong_down"):
        close_market_strict("CHOP_EXIT")
        return
    
    votes, rs = council_exhaustion_votes(df, ind, info, zones, trend)
    if votes >= EXH_VOTES_NEEDED:
        close_market_strict("SCM_EXHAUSTION: " + ",".join(rs))
        return
    
    xp = xprotect_signal(df, ind, info)
    if side == "long" and xp["explode_down"]:
        close_market_strict("XPROTECT_LONG_EXPLODE_DOWN")
        return
    if side == "short" and xp["explode_up"]:
        close_market_strict("XPROTECT_SHORT_EXPLODE_UP")
        return
    
    adx = float(ind.get("adx") or 0.0)
    hyst = _near_bps(info["price"], info["filter"])
    if opp and adx >= BREAK_ADX_MIN and hyst >= EXH_HYST_MIN_BPS:
        close_market_strict("OPPOSITE_RF_CONFIRMED")
        return

# =================== الدخول المحسن ===================
def enhanced_open_market(side, qty, price, strength, reason, df, ind):
    """فتح صفقة محسن مع إدارة متكاملة"""
    global ENTRY_IN_PROGRESS, _last_entry_attempt_ts, PENDING_OPEN, LAST_SIGNAL_USED
    
    if _now() - _last_entry_attempt_ts < ENTRY_GUARD_WINDOW_SEC:
        print(colored("⏸️ entry guard window — skip", "yellow"))
        return False
    
    if ENTRY_LOCK.locked() or ENTRY_IN_PROGRESS or PENDING_OPEN:
        print(colored("⏸️ entry in progress/pending — skip", "yellow"))
        return False
    
    with ENTRY_LOCK:
        ENTRY_IN_PROGRESS = True
        PENDING_OPEN = True
        
        try:
            ex_qty, ex_side, _ = _read_position()
            if ex_qty and ex_qty > 0:
                print(colored(f"⛔ exchange already has position ({ex_side}) — skip open", "red"))
                return False
            
            _cancel_symbol_orders()
            bal = balance_usdt()
            px = float(price or price_now() or 0.0)
            
            # حساب حجم المركز بناء على قوة الإشارة
            q_total = calculate_position_size(bal, px, strength)
            
            if q_total <= 0 or (LOT_MIN and q_total < LOT_MIN):
                print(colored(f"❌ skip open (qty too small) — bal={fmt(bal,2)} px={fmt(px)} q={q_total}", "red"))
                return False
            
            sp = orderbook_spread_bps()
            if sp is not None and sp > SPREAD_HARD_BPS:
                print(colored(f"⛔ hard spread guard: {fmt(sp,2)}bps > {SPREAD_HARD_BPS}", "red"))
                return False
            
            link = _order_link("ENT")
            if MODE_LIVE:
                ex.create_order(SYMBOL, "market", side, q_total, None, {**_params_open(side), "orderLinkId": link})
            else:
                print(colored(f"[PAPER] create_order market {side} {q_total}", "cyan"))
            
            time.sleep(0.45)
            cur_qty, cur_side, cur_entry = _read_position()
            
            if not cur_qty or cur_qty <= 0:
                print(colored("❌ open failed — no position filled", "red"))
                return False
            
            expected_side = "long" if side == "buy" else "short"
            if cur_side not in ("long", "short") or cur_side != expected_side:
                print(colored(f"❌ side mismatch after open (expected {expected_side}, got {cur_side}) — strict close", "red"))
                close_market_strict("SIDE_MISMATCH_AFTER_OPEN")
                return False
            
            # إعداد نظام إدارة الصفقة
            atr = float(ind.get("atr", 0))
            setup_trade_management(float(cur_entry), atr, cur_side, strength)
            
            STATE.update({
                "open": True, "side": cur_side, "entry": float(cur_entry),
                "qty": safe_qty(cur_qty), "remaining_size": safe_qty(cur_qty),
                "pnl": 0.0, "bars": 0, "trail": None,
                "hp_pct": 0.0, "strength": float(strength),
                "peak_adx": 0.0, "rsi_peak": 50.0, "rsi_trough": 50.0,
                "peak_price": float(cur_entry), "trough_price": float(cur_entry),
                "opp_rf_count": 0, "chop_flag": False
            })
            
            TRADE_TIMES.append(time.time())
            _last_entry_attempt_ts = _now()
            
            # تحديث آخر إشارة مستخدمة
            LAST_SIGNAL_USED.update({
                "side": side,
                "bar_ts": _last_closed_bar_ts(fetch_ohlcv()),
                "src": reason.split(" ")[0] if reason else "unknown",
                "strength": float(strength)
            })
            
            print(colored(
                f"🚀 OPEN {('🟩 LONG' if cur_side=='long' else '🟥 SHORT')} | "
                f"qty={fmt(STATE['qty'],4)} @ {fmt(STATE['entry'])} | "
                f"strength={fmt(strength,2)} | reason={reason}",
                "green" if cur_side=='long' else 'red'
            ))
            
            # تحليل الشمعة الحالية
            candle_analysis = analyze_candle_strength(df, ind)
            print(colored(f"📊 قوة الشمعة: {fmt(candle_analysis['strength'],2)} | زخم: {fmt(candle_analysis['momentum'],2)}%", "cyan"))
            
            logging.info(f"OPEN {cur_side} qty={STATE['qty']} entry={STATE['entry']} strength={strength} reason={reason}")
            return True
            
        except Exception as e:
            print(colored(f"❌ open error: {e}", "red"))
            logging.error(f"open_market error: {e}", exc_info=True)
            return False
        finally:
            ENTRY_IN_PROGRESS = False
            PENDING_OPEN = False

# =================== الدورة الرئيسية المحسنة ===================
def enhanced_trade_loop():
    """الدورة الرئيسية المحسنة للتداول"""
    global LAST_CLOSE_TS, LAST_DECISION_BAR_TS, _last_entry_attempt_ts
    global POST_CHOP_BLOCK_ACTIVE, POST_CHOP_BLOCK_UNTIL_BAR, LAST_CLOSE_BAR_TS

    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            reconcile_state()

            info, ind, zones, candidates, trend, plan = evaluate_all(df)

            spread_bps = orderbook_spread_bps()
            reason = None
            
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"

            since_last_close = time.time() - LAST_CLOSE_TS
            if reason is None and since_last_close < max(COOLDOWN_SEC, REENTRY_COOLDOWN_SEC):
                remain = int(max(COOLDOWN_SEC, REENTRY_COOLDOWN_SEC) - since_last_close)
                reason = f"cooldown {remain}s"

            while TRADE_TIMES and time.time() - TRADE_TIMES[0] > 3600:
                TRADE_TIMES.popleft()
            if reason is None and len(TRADE_TIMES) >= MAX_TRADES_PER_HOUR:
                reason = "rate-limit: too many trades this hour"

            # اختيار أفضل دخول محسن
            current_bar_ts = _last_closed_bar_ts(df)
            if reason is None and candidates:
                best = enhanced_entry_decision(candidates, df, ind, info)
                
                if best and LAST_SIGNAL_USED["side"] == best["side"] and \
                   LAST_SIGNAL_USED["bar_ts"] == current_bar_ts and \
                   LAST_SIGNAL_USED["src"] == best["src"]:
                    reason = f"⛔ same signal already used this bar ({best['side']} from {best['src']})"
                    best = None
            else:
                best = None

            if STATE["open"] and px:
                STATE["pnl"] = (px - STATE["entry"]) * STATE["qty"] if STATE["side"] == "long" else (STATE["entry"] - px) * STATE["qty"]
                STATE["hp_pct"] = max(STATE.get("hp_pct", 0.0), (px - STATE["entry"]) / STATE["entry"] * 100.0 * (1 if STATE["side"] == "long" else -1))
                _update_trend_state(ind, {"price": px, **info})

            # استخدام الإدارة المحسنة للصفقة
            enhanced_manage_position(df, ind, {"price": px or info["price"], **info}, zones, trend)

            bar_ts = _last_closed_bar_ts(df)
            if POST_CHOP_BLOCK_ACTIVE and bar_ts >= POST_CHOP_BLOCK_UNTIL_BAR:
                if POST_CHOP_REQUIRE_RF and not (info.get("long") or info.get("short")):
                    pass
                else:
                    POST_CHOP_BLOCK_ACTIVE = False

            if not STATE["open"] and best and reason is None:
                if _now() - _last_entry_attempt_ts < max(ENTRY_GUARD_WINDOW_SEC, MIN_SIGNAL_AGE_SEC):
                    reason = "entry guard window"
                else:
                    adx_now = float(ind.get("adx") or 0.0)
                    
                    # شروط الدخول المشددة
                    volume_analysis = analyze_volume(df)
                    entry_conditions_met = (
                        best.get("final_strength", 0) >= 4.5 and
                        adx_now >= 20 and
                        volume_analysis["volume_ok"] and
                        abs(float(ind.get("plus_di") or 0) - float(ind.get("minus_di") or 0)) >= 4
                    )
                    
                    if entry_conditions_met:
                        ok = enhanced_open_market(
                            "buy" if best["side"] == "buy" else "sell",
                            best.get("qty", 0),
                            px or info["price"],
                            best["final_strength"],
                            f"ELITE_ENHANCED: {best['reason']} | boosts: {best.get('boost_factors', 0)}",
                            df, ind
                        )
                        _last_entry_attempt_ts = _now()
                        if not ok:
                            reason = "open failed (elite enhanced conditions)"
                    else:
                        reason = f"elite enhanced conditions not met: strength={best.get('final_strength'):.1f} adx={adx_now:.1f} vol_ok={volume_analysis['volume_ok']}"

            # عرض معلومات إضافية
            enhanced_pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, zones, reason, df)

            if len(df) >= 2 and int(df["time"].iloc[-1]) != int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"] += 1

            time.sleep(NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP)

        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

def enhanced_pretty_snapshot(bal, info, ind, spread_bps, zones, reason=None, df=None):
    """عرض محسن للمعلومات"""
    left_s = time_to_candle_close(df) if df is not None else 0
    
    print(colored("═" * 120, "cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", "cyan"))
    print(colored("═" * 120, "cyan"))
    
    # معلومات السوق
    print("📈 MARKET ANALYSIS")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))} | spread={fmt(spread_bps,2)} bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
    print(f"   📊 MACD={fmt(ind.get('macd_hist'))}  VWAP={fmt(ind.get('vwap'))}  Delta={fmt(ind.get('delta_vol'))}")
    
    # تحليل الحجم والشموع
    volume_analysis = analyze_volume(df)
    candle_strength = analyze_candle_strength(df, ind)
    candle_patterns = detect_strong_candle_patterns(df)
    
    print(f"   🔊 Volume: {volume_analysis['volume_trend']} (x{fmt(volume_analysis['volume_ratio'],2)})")
    print(f"   🕯️ Candle: strength={fmt(candle_strength['strength'],2)} momentum={fmt(candle_strength['momentum'],2)}%")
    
    # معلومات التداول
    print(f"   🧠 {STATE.get('scm_line','')}")
    print(f"   🧊 CHOP={STATE.get('chop_flag', False)} | POST_CHOP_BLOCK={POST_CHOP_BLOCK_ACTIVE}")
    print(f"   🧭 PLAN={STATE.get('plan','SIT_OUT')} • reasons={STATE.get('plan_reasons',[])}")
    print(f"   🗳️ votes: BUY={STATE.get('votes_b',0)}({fmt(STATE.get('score_b',0),2)}) SELL={STATE.get('votes_s',0)}({fmt(STATE.get('score_s',0),2)})")
    print(f"   ⏱️ closes_in ≈ {left_s}s")
    
    print("\n🧭 POSITION & MANAGEMENT")
    bal_line = f"Balance={fmt(bal,2)} Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x CompoundPnL={fmt(compound_pnl)} Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(colored(f"   {bal_line}", "yellow"))
    
    if STATE["open"]:
        lamp = '🟩 LONG' if STATE['side'] == 'long' else '🟥 SHORT'
        tm = STATE["trade_management"]
        
        print(f"   {lamp} Entry={fmt(STATE['entry'])} Qty={fmt(STATE['qty'],4)} Remaining={fmt(STATE.get('remaining_size', STATE['qty']),4)}")
        print(f"   📊 PnL={fmt(STATE['pnl'],2)} HP={fmt(STATE['hp_pct'],2)}% Bars={STATE['bars']}")
        print(f"   🛡️ Stop={fmt(tm['current_stop'])} Trail={'✅' if tm['trailing_active'] else '❌'} BreakEven={'✅' if tm['break_even_moved'] else '❌'}")
        print(f"   🎯 Targets: {len(tm['targets_hit'])}/{len(TAKE_PROFIT_LEVELS)} hit")
    else:
        print("   ⚪ FLAT")
    
    if reason:
        print(colored(f"   ℹ️ reason: {reason}", "white"))
    
    print(colored("═" * 120, "cyan"))

# =================== التشغيل ===================
if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • {SYMBOL} • {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x • ADX(min RF)={ADX_ENTRY_MIN}", "yellow"))
    print(colored("🎯 BOT PRO PLUS FEATURES: Enhanced Trade Management • Multi-Take Profit • Advanced Candle Analysis", "green"))
    print(colored("🎯 إدارة صفقات محترفة • جني أرباح متعدد • تحليل شموع متقدم • وقف خسارة ديناميكي", "green"))
    
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    
    import threading as _t
    _t.Thread(target=enhanced_trade_loop, daemon=True).start()
    _t.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
