# -*- coding: utf-8 -*-
"""
BYBIT — SUI Perp Bot (RF Closed + Strong Council + Smart Exec Locks + Chop/Accumulation Exit)
• ENV فقط: BYBIT_API_KEY, BYBIT_API_SECRET, SELF_URL/RENDER_EXTERNAL_URL, PORT
• دخول RF على الشمعة المغلقة فقط
• مجلس الإدارة لا يدخل إلا لو التصويت قوي ومؤكد
• قرار تداول واحد لكل شمعة مغلقة
• Locks + orderLinkId + cancel_all_orders قبل أي تنفيذ
• غلق صارم متدرّج يعالج -110017 (reduceOnly)
• مصالحة حالة البوت مع المنصّة + جني ربح للفتائل/الشموع الكبيرة + X-Protect
• NEW: منع فتح صفقتين لنفس الاتجاه “من المصرف” (تأكد منصّة + Pending)
• NEW: Chop/Accumulation Detector → غلق صارم مبكر على ربح بسيط + سياسة انتظار ذكية
"""

import os, time, math, random, signal, sys, traceback, logging, uuid, threading
from logging.handlers import RotatingFileHandler
from datetime import datetime
from collections import deque
from decimal import Decimal, ROUND_DOWN, InvalidOperation

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

# =================== SETTINGS ===================
SYMBOL        = "SUI/USDT:USDT"
INTERVAL      = "15m"

LEVERAGE      = 10
RISK_ALLOC    = 0.60
POSITION_MODE = "oneway"  # positionIdx=0

# RF (شمعة مُغلقة فقط)
RF_SOURCE   = "close"
RF_PERIOD   = 20
RF_MULT     = 3.5
RF_HYST_BPS = 6.0

# مؤشرات
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# حمايات عامة
ADX_ENTRY_MIN   = 17.0
MAX_SPREAD_BPS  = 8.0
SPREAD_HARD_BPS = 15.0
COOLDOWN_SEC    = 90
MAX_TRADES_PER_HOUR = 6

# صناديق/مناطق
LEVEL_NEAR_BPS  = 12.0
WICK_RATIO_MIN  = 0.55

# SCM — Breakout/Retest/Trap/Trend
BREAK_HYST_BPS     = 10.0
BREAK_ADX_MIN      = 22.0
BREAK_DI_MARGIN    = 5.0
BREAK_BODY_ATR_MIN = 0.60

LIQ_EQ_LOOKBACK    = 30
LIQ_EQ_TOL_BPS     = 6.0
SWEEP_WICK_RATIO   = 0.55
DISP_BODY_ATR_MIN  = 0.75
RETEST_MAX_BARS    = 6
TRAP_CLOSE_BACK_BPS= 8.0

# ترند قوي / Debounce RF
TREND_STRONG_ADX   = 28.0
TREND_STRONG_DI_M  = 8.0
OPP_RF_DEBOUNCE    = 2

# إدارة الصفقة
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6

# Wick/Big Candle harvest
WICK_TAKE_MIN_PCT   = 0.40
WICK_BIG_RATIO      = 0.62
BODY_BIG_ATR_MULT   = 1.10

# خروج نهاية موجة (SCM Exhaustion)
EXH_MIN_PNL_PCT   = 0.35
EXH_ADX_DROP      = 6.0
EXH_ADX_MIN       = 18.0
EXH_RSI_PULLBACK  = 7.0
EXH_WICK_RATIO    = 0.60
EXH_HYST_MIN_BPS  = 8.0
EXH_BOS_LOOKBACK  = 6
EXH_VOTES_NEEDED  = 3

# تصويت الدخول (قوي فقط)
COUNCIL_ENTRY_VOTES_MIN  = 5
COUNCIL_STRONG_SCORE_MIN = 3.5

# Slippage / Exec
MAX_SLIP_OPEN_BPS   = 25.0
MAX_SLIP_CLOSE_BPS  = 35.0

# X-Protect
VEI_LEN_BASE      = 50
VEI_EXPLODE_MULT  = 2.2
VEI_FILTER_BPS    = 12.0
VEI_ADX_MIN       = 18.0
VEI_VOL_VOTE      = 1

# إيقاع
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# =================== NEW — Chop/Accumulation Exit ===================
# تعريف "التذبذب/التجميع" = ADX منخفض + انكماش ATR% + نطاق ضيق + (اختياري) باند بولنجر ضيق
BB_LEN                 = 20
CHOP_ADX_MAX           = 16.0
CHOP_LOOKBACK          = 120     # للحسابات الإحصائية
CHOP_ATR_PCT_FRACTION  = 0.65    # ATR% الحالي ≤ 65% من وسيط ATR% (ضغط تقلبات)
CHOP_BB_WIDTH_PCT_MAX  = 1.10    # (upper-lower)/middle *100 ≤ 1.1%
CHOP_RANGE_BARS        = 24      # آخر 24 شمعة: النطاق ضيق؟
CHOP_RANGE_BPS_MAX     = 60.0    # (maxH-minL)/mid ≤ 60bps
CHOP_MIN_PNL_PCT       = 0.20    # لو فيه ربح ≥ 0.20% نخرج فورًا غلق صارم
# سياسة ما بعد الخروج بسبب التذبذب:
POST_CHOP_WAIT_BARS    = 2       # انتظر عدد شموع مغلقة
POST_CHOP_REQUIRE_RF   = True    # مطلوب RF مغلق للدخول مجددًا… أو مجلس استثنائي قوي جدًا (اختراق منطقة)

# سياسة عامة إضافية لمنع الدخول المتكرر
MIN_REENTRY_BARS       = 1       # بعد أي غلق: انتظر شمعة مغلقة واحدة على الأقل

# =================== Logging ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h,"baseFilename","").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready", "cyan"))
setup_file_logging()

# =================== Exchange ===================
def make_ex():
    return ccxt.bybit({
        "apiKey": API_KEY,
        "secret": API_SEC,
        "enableRateLimit": True,
        "timeout": 20000,
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

# =================== Locks / Guards ===================
ENTRY_LOCK = threading.Lock()
CLOSE_LOCK = threading.Lock()
ENTRY_IN_PROGRESS = False
CLOSE_IN_PROGRESS = False
PENDING_OPEN = False       # NEW: يمنع Double-Open لحظي حتى قبل انعكاس حالة المنصّة
ENTRY_GUARD_WINDOW_SEC = 6
CLOSE_GUARD_WINDOW_SEC = 3
_last_entry_attempt_ts = 0.0
_last_close_attempt_ts = 0.0
LAST_DECISION_BAR_TS = 0  # قرار واحد لكل شمعة مغلقة

# سياسة انتظارات
LAST_CLOSE_BAR_TS = 0
POST_CHOP_BLOCK_ACTIVE = False
POST_CHOP_BLOCK_UNTIL_BAR = 0

def _now(): return time.time()
def _order_link(prefix="ORD"): return f"{prefix}-{uuid.uuid4().hex[:18]}"

# =================== Helpers / State ===================
LAST_CLOSE_TS = 0
TRADE_TIMES = deque(maxlen=10)
compound_pnl = 0.0

STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None,
    "hp_pct": 0.0, "strength": 0.0,
    "peak_adx": 0.0, "rsi_peak": 50.0, "rsi_trough": 50.0,
    "peak_price": 0.0, "trough_price": 0.0,
    "opp_rf_count": 0,
    "scm_line": "",
    "chop_flag": False
}

def _norm_sym(s: str) -> str: return (s or "").replace("/", "").replace(":", "").upper()
def _sym_match(a: str, b: str) -> bool:
    A, B = _norm_sym(a), _norm_sym(b); return A == B or A in B or B in A

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

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

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
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
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

# ============ أدوات حماية الانزلاق ============
def _best_bid_ask():
    ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
    bid = ob["bids"][0][0] if ob["bids"] else None
    ask = ob["asks"][0][0] if ob["asks"] else None
    return bid, ask

def _price_band(side:str, px:float, max_bps:float):
    if px is None: return None
    if side == "buy":  return px * (1 + max_bps/10000.0)
    else:              return px * (1 - max_bps/10000.0)

# =================== RF (CLOSED) ===================
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 3:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)
    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))
    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wilder_ema(plus_dm, ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wilder_ema(minus_dm, ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wilder_ema(dx, ADX_LEN)
    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i])
    }

def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n); wper = (n*2)-1
    return _ema(avrng, wper) * qty
def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt=pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def rf_signal_closed(df: pd.DataFrame):
    if len(df) < RF_PERIOD + 3:
        i = -2 if len(df) >= 2 else -1
        price = float(df["close"].iloc[i]) if len(df) else None
        t     = int(df["time"].iloc[i]) if len(df) else int(time.time()*1000)
        return {"time": t, "price": price or 0.0, "long": False, "short": False,
                "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0}
    d = df.iloc[:-1].copy()  # ← الشمعة المغلقة فقط
    src = d[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    p_prev = float(src.iloc[-1]); f_prev = float(filt.iloc[-1])
    long_sig  = (p_prev > f_prev and _bps(p_prev, f_prev) >= RF_HYST_BPS)
    short_sig = (p_prev < f_prev and _bps(p_prev, f_prev) >= RF_HYST_BPS)
    return {"time": int(d["time"].iloc[-1]), "price": p_prev, "long": bool(long_sig),
            "short": bool(short_sig), "filter": f_prev,
            "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])}

# =================== Zones / Liquidity ===================
def _find_swings(df: pd.DataFrame, left:int=2, right:int=2):
    if len(df) < left+right+3: return None, None
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    ph=[None]*len(df); pl=[None]*len(df)
    for i in range(left, len(df)-right):
        if all(h[i] >= h[j] for j in range(i-left, i+right+1)): ph[i]=h[i]
        if all(l[i] <= l[j] for j in range(i-left, i+right+1)): pl[i]=l[i]
    return ph, pl

def detect_zones(df: pd.DataFrame):
    try:
        d = df.iloc[:-1] if len(df)>=2 else df.copy()
        ph, pl = _find_swings(d, 2, 2)
        highs = [p for p in ph if p is not None][-15:]
        lows  = [p for p in pl if p is not None][-15:]
        sup=None; dem=None
        if highs:
            top=max(highs); bot=max(highs) - (max(highs)-min(highs))*0.25
            sup={"side":"supply","top":float(top),"bot":float(bot)}
        if lows:
            bot=min(lows); top=min(lows) + (max(lows)-min(lows))*0.25 if len(lows)>1 else bot*1.002
            dem={"side":"demand","top":float(top),"bot":float(bot)}
        return {"supply":sup, "demand":dem}
    except Exception:
        return {"supply":None, "demand":None}

def find_equal_highs_lows(df: pd.DataFrame):
    if len(df) < LIQ_EQ_LOOKBACK+2: return None, None
    d = df.iloc[-(LIQ_EQ_LOOKBACK+1):-1]
    highs = d["high"].astype(float).values
    lows  = d["low"].astype(float).values
    eh = max(highs); el = min(lows)
    def _cluster(vals, target, tol_bps):
        cnt = sum(1 for v in vals if abs((v-target)/target)*10000.0 <= tol_bps)
        return cnt>=3
    eqh_ok = _cluster(highs, eh, LIQ_EQ_TOL_BPS)
    eql_ok = _cluster(lows,  el, LIQ_EQ_TOL_BPS)
    return (eh if eqh_ok else None), (el if eql_ok else None)

def detect_sweep(df: pd.DataFrame, eqh, eql):
    if len(df) < 2: return {"sweep_up":False,"sweep_down":False}
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
    l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
    if eqh:
        broke_up = h>eqh and (h-c) >= SWEEP_WICK_RATIO*(h-l)
        if broke_up and c<eqh: return {"sweep_up":True,"sweep_down":False}
    if eql:
        broke_dn = l<eql and (c-l) >= SWEEP_WICK_RATIO*(h-l)
        if broke_dn and c>eql: return {"sweep_up":False,"sweep_down":True}
    return {"sweep_up":False,"sweep_down":False}

# =================== X-Protect ===================
def xprotect_signal(df: pd.DataFrame, ind: dict, info: dict):
    if len(df) < VEI_LEN_BASE + 5:
        return {"explode_up":False,"explode_down":False,"vei":1.0,"why":"warmup"}
    closes = df["close"].astype(float)
    highs  = df["high"].astype(float)
    lows   = df["low"].astype(float)
    tr = pd.concat([(highs-lows).abs(), (highs-closes.shift(1)).abs(), (lows-closes.shift(1)).abs()], axis=1).max(axis=1)
    atr_series = wilder_ema(tr, ATR_LEN)
    atr_pct_series = (atr_series / closes.replace(0,1e-12)) * 100.0
    base = atr_pct_series.ewm(span=VEI_LEN_BASE, adjust=False).mean()
    vei = float((atr_pct_series.iloc[-1] / max(base.iloc[-1], 1e-9)))
    adx = float(ind.get("adx") or 0.0)
    filt = float(info.get("filter") or closes.iloc[-1])
    px   = float(info.get("price")  or closes.iloc[-1])
    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    away = _bps(px, filt)
    explode = (vei >= VEI_EXPLODE_MULT and adx >= VEI_ADX_MIN and away >= VEI_FILTER_BPS)
    o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
    up = (c>o); dn = (c<o)
    return {"explode_up": bool(explode and up),
            "explode_down": bool(explode and dn),
            "vei": vei, "why": f"vei={vei:.2f} adx={adx:.1f} away={away:.1f}bps"}

# =================== SCM (التصويت) ===================
def displacement_bar(_row_open: float, _row_close: float, atr: float, side: str) -> bool:
    if atr <= 0: 
        return False
    body = abs(_row_close - _row_open)
    if side == "buy":
        return (_row_close > _row_open) and (body >= DISP_BODY_ATR_MIN * atr)
    else:
        return (_row_close < _row_open) and (body >= DISP_BODY_ATR_MIN * atr)

def retest_happened(history_df: pd.DataFrame, zones: dict, side: str) -> bool:
    try:
        if len(history_df) < RETEST_MAX_BARS + 2: return False
        d = history_df.iloc[-(RETEST_MAX_BARS+1):-1]
        sup, dem = zones.get("supply"), zones.get("demand")
        closes = d["close"].astype(float).values
        if side == "buy" and sup:
            mid = (sup["top"] + sup["bot"]) / 2.0
            return any((px >= sup["bot"] and px <= sup["top"]) or (px >= mid) for px in closes)
        if side == "sell" and dem:
            mid = (dem["top"] + dem["bot"]) / 2.0
            return any((px <= dem["top"] and px >= dem["bot"]) or (px <= mid) for px in closes)
        return False
    except Exception:
        return False

def trap_detect_row(o: float, c: float, zones: dict, side: str) -> bool:
    sup, dem = zones.get("supply"), zones.get("demand")
    def _bps(a,b): 
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    if side=="buy" and sup and c < sup["top"] and _bps(o, sup["top"]) >= BREAK_HYST_BPS and _bps(c, sup["top"]) <= TRAP_CLOSE_BACK_BPS:
        return True
    if side=="sell" and dem and c > dem["bot"] and _bps(o, dem["bot"]) >= BREAK_HYST_BPS and _bps(c, dem["bot"]) <= TRAP_CLOSE_BACK_BPS:
        return True
    return False

def trend_context(ind: dict):
    adx=float(ind.get("adx") or 0.0)
    pdi=float(ind.get("plus_di") or 0.0)
    mdi=float(ind.get("minus_di") or 0.0)
    if adx>=TREND_STRONG_ADX and abs(pdi-mdi)>=TREND_STRONG_DI_M:
        return "strong_up" if pdi>mdi else "strong_down"
    if pdi>mdi: return "up"
    if mdi>pdi: return "down"
    return "sideways"

def council_scm_votes(df, ind, info, zones):
    # استخدم آخر شمعة مغلقة فقط لكل ما يمكن
    d = df.iloc[:-1] if len(df) >= 2 else df
    if len(d) < 1:
        return 0,[],0,[],0.0,0.0,"SCM | warmup", "sideways"

    o = float(d["open"].iloc[-1]); c = float(d["close"].iloc[-1])
    h = float(d["high"].iloc[-1]); l = float(d["low"].iloc[-1])

    reasons_b=[]; reasons_s=[]; b=s=0; score_b=0.0; score_s=0.0
    trend = trend_context(ind)

    atr=float(ind.get("atr") or 0.0); adx=float(ind.get("adx") or 0.0)
    pdi=float(ind.get("plus_di") or 0.0); mdi=float(ind.get("minus_di") or 0.0)
    body=abs(c-o)

    def _bps(a,b): 
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0

    sup, dem = zones.get("supply"), zones.get("demand")

    boxes="—"; strong_box_b=False; strong_box_s=False
    if sup and c>sup["top"] and _bps(c,sup["top"])>=BREAK_HYST_BPS and body>=BREAK_BODY_ATR_MIN*max(atr,1e-9) and adx>=BREAK_ADX_MIN and (pdi>=mdi+BREAK_DI_MARGIN):
        b+=2; score_b+=1.6; reasons_b.append("breakout@supply +2"); boxes="↑sup"; strong_box_b=True
    if dem and c<dem["bot"] and _bps(c,dem["bot"])>=BREAK_HYST_BPS and body>=BREAK_BODY_ATR_MIN*max(atr,1e-9) and adx>=BREAK_ADX_MIN and (mdi>=pdi+BREAK_DI_MARGIN):
        s+=2; score_s+=1.6; reasons_s.append("breakout@demand +2"); boxes="↓dem" if boxes=="—" else boxes+"|↓dem"; strong_box_s=True

    eqh,eql = find_equal_highs_lows(df)
    sw = detect_sweep(df, eqh, eql)
    liquidity="—"
    if sw["sweep_up"]:
        s+=1; score_s+=0.6; reasons_s.append("sweep_up"); liquidity="sweep_up"
    if sw["sweep_down"]:
        b+=1; score_b+=0.6; reasons_b.append("sweep_down"); liquidity=("sweep_down" if liquidity=="—" else liquidity+"|sweep_down")
    if eqh: liquidity = ("EQH" if liquidity=="—" else liquidity+"|EQH")
    if eql: liquidity = ("EQL" if liquidity=="—" else liquidity+"|EQL")

    displacement="—"
    if displacement_bar(o, c, atr, "buy"):   b+=1; score_b+=0.7; reasons_b.append("displacement+"); displacement="up"
    if displacement_bar(o, c, atr, "sell"):  s+=1; score_s+=0.7; reasons_s.append("displacement-"); displacement=("down" if displacement=="—" else displacement+"|down")

    retest="—"
    if retest_happened(df, zones, "buy"):  b+=1; score_b+=0.5; reasons_b.append("retest@supbreak"); retest="up"
    if retest_happened(df, zones, "sell"): s+=1; score_s+=0.5; reasons_s.append("retest@dembreak"); retest=("down" if retest=="—" else retest+"|down")

    trap="—"
    if trap_detect_row(o, c, zones, "buy"):   b+=1; score_b+=0.6; reasons_b.append("trap@sup"); trap="bull"
    if trap_detect_row(o, c, zones, "sell"):  s+=1; score_s+=0.6; reasons_s.append("trap@dem"); trap=("bear" if trap=="—" else trap+"|bear")

    # RF كمساند (مبني أصلًا على الشمعة المغلقة)
    if info.get("long"):  b+=1; score_b+=0.5; reasons_b.append("rf_long")
    if info.get("short"): s+=1; score_s+=0.5; reasons_s.append("rf_short")

    if pdi>mdi and adx>=18: b+=1; score_b+=0.5; reasons_b.append("DI+>DI- & ADX")
    if mdi>pdi and adx>=18: s+=1; score_s+=0.5; reasons_s.append("DI->DI+ & ADX")

    xp = xprotect_signal(df, ind, info)
    if xp["explode_up"]:   b += VEI_VOL_VOTE; score_b += 0.5; reasons_b.append(f"xprotect_up {xp['why']}")
    if xp["explode_down"]: s += VEI_VOL_VOTE; score_s += 0.5; reasons_s.append(f"xprotect_down {xp['why']}")

    score_b += b/4.0; score_s += s/4.0
    scm_line = f"SCM | {trend} | {boxes} | {liquidity} | {displacement} | {retest} | {trap} | votes(b={b},s={s})"
    return (b,reasons_b,s,reasons_s,score_b,score_s,scm_line,trend,
            strong_box_b, strong_box_s)

def council_entry(df, ind, info, zones):
    b,b_r,s,s_r,score_b,score_s,scm_line,trend,box_b,box_s = council_scm_votes(df, ind, info, zones)
    STATE["scm_line"] = scm_line
    candidates=[]
    # Council قوي فقط
    if b >= COUNCIL_ENTRY_VOTES_MIN and score_b >= COUNCIL_STRONG_SCORE_MIN:
        candidates.append({"side":"buy","score":score_b,"votes":b,"reason":f"Council BUY {b} :: {b_r}",
                           "trend":trend,"src":"council","box_break":box_b})
    if s >= COUNCIL_ENTRY_VOTES_MIN and score_s >= COUNCIL_STRONG_SCORE_MIN:
        candidates.append({"side":"sell","score":score_s,"votes":s,"reason":f"Council SELL {s} :: {s_r}",
                           "trend":trend,"src":"council","box_break":box_s})
    # RF (شمعة مغلقة)
    if info.get("long"):
        candidates.append({"side":"buy","score":1.0,"votes":0,"reason":"RF_LONG (closed)","trend":trend,"src":"rf","box_break":False})
    if info.get("short"):
        candidates.append({"side":"sell","score":1.0,"votes":0,"reason":"RF_SHORT (closed)","trend":trend,"src":"rf","box_break":False})
    # ترتيب: council ثم RF
    candidates.sort(key=lambda x: (x["src"]!="council", -x["score"]))
    return candidates, trend

def choose_best_entry(candidates, ind):
    if not candidates: return None
    # لو فيه بلوك ما بعد التذبذب: السماح فقط بـ RF، أو Council استثنائي مع اختراق صندوق + أصوات أعلى
    if POST_CHOP_BLOCK_ACTIVE:
        strong_overrides = [c for c in candidates if c["src"]=="council" and c.get("box_break") and c.get("votes",0) >= (COUNCIL_ENTRY_VOTES_MIN+2) and c.get("score",0)>= (COUNCIL_STRONG_SCORE_MIN+0.8)]
        rf_only          = [c for c in candidates if c["src"]=="rf"]
        return strong_overrides[0] if strong_overrides else (rf_only[0] if rf_only else None)
    # عادي: اختَر Council أولًا
    for c in candidates:
        if c["src"]=="council": return c
    return next((c for c in candidates if c["src"]=="rf"), None)

# =================== تنفيذ (فتح/غلق) ===================
def _params_open(side):
    return {"positionSide": "BOTH", "reduceOnly": False, "positionIdx": 0}

def _params_close():
    return {"positionSide": "BOTH", "reduceOnly": True, "positionIdx": 0}

def _bybit_reduceonly_reject(err: Exception) -> bool:
    m = str(err).lower()
    return ("-110017" in m) or ("reduce-only order has been rejected" in m)

def _cancel_symbol_orders():
    try:
        if MODE_LIVE:
            ex.cancel_all_orders(SYMBOL)
            print(colored("🧹 canceled all open orders for symbol", "yellow"))
    except Exception as e:
        print(colored(f"⚠️ cancel_all_orders warn: {e}", "yellow"))

def _read_position():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym = (p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if not _sym_match(sym, SYMBOL): continue
            qty=None
            for key in ("contracts","size","positionAmt"):
                v = p.get(key) or p.get("info",{}).get(key)
                if v is not None:
                    try: qty = abs(float(v)); break
                    except Exception: pass
            if not qty or qty<=0: continue
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0) or 0.0
            side_raw = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            cost = float(p.get("cost") or p.get("info",{}).get("positionValue") or 0)
            side = "long" if ("long" in side_raw or cost>=0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position error: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    if not balance or balance <= 0 or not price or price <= 0:
        return 0.0
    equity = float(balance)
    px = max(float(price), 1e-9)
    buffer = 0.97
    notional = equity * RISK_ALLOC * LEVERAGE * buffer
    raw_qty = notional / px
    q_norm = safe_qty(raw_qty)
    if q_norm <= 0:
        lot_min = LOT_MIN or 0.1
        need = (lot_min * px) / (LEVERAGE * RISK_ALLOC * buffer)
        print(colored(f"⚠️ balance {fmt(balance,2)} too small — need ≥ {fmt(need,2)} USDT to meet min lot {lot_min}", "yellow"))
        return 0.0
    return q_norm

def open_market(side, qty, price, strength, reason):
    """
    فتح صفقة آمن “من المنصّة”:
    - منع التوازي + نافذة حماية زمنية + PENDING_OPEN
    - cancel_all_orders أولًا
    - Market واحد + orderLinkId
    - مصالحة بعد التنفيذ + تحقق منصّة يمنع Double-Open في نفس الاتجاه
    """
    global ENTRY_IN_PROGRESS, _last_entry_attempt_ts, PENDING_OPEN

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
            # تأكد من عدم وجود مركز فعلي على المنصّة
            ex_qty, ex_side, _ = _read_position()
            if ex_qty and ex_qty > 0:
                print(colored(f"⛔ exchange already has position ({ex_side}) — skip open", "red"))
                return False

            _cancel_symbol_orders()

            bal = balance_usdt()
            px = float(price or price_now() or 0.0)
            q_total = safe_qty(min(qty, compute_size(bal, px)))
            if q_total <= 0 or (LOT_MIN and q_total < LOT_MIN):
                print(colored(f"❌ skip open (qty too small) — bal={fmt(bal,2)} px={fmt(px)} q={fmt(q_total,4)}", "red"))
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

            STATE.update({
                "open": True, "side": cur_side, "entry": float(cur_entry),
                "qty": safe_qty(cur_qty), "pnl": 0.0, "bars": 0, "trail": None,
                "hp_pct": 0.0, "strength": float(strength),
                "peak_adx": 0.0, "rsi_peak": 50.0, "rsi_trough": 50.0,
                "peak_price": float(cur_entry), "trough_price": float(cur_entry),
                "opp_rf_count": 0, "chop_flag": False
            })
            TRADE_TIMES.append(time.time())
            _last_entry_attempt_ts = _now()

            print(colored(
                f"🚀 OPEN {('🟩 LONG' if cur_side=='long' else '🟥 SHORT')} "
                f"qty={fmt(STATE['qty'],4)} @ {fmt(STATE['entry'])} | strength={fmt(strength,2)} | {reason}",
                "green" if cur_side=='long' else 'red'
            ))
            logging.info(f"OPEN {cur_side} qty={STATE['qty']} entry={STATE['entry']} strength={strength} reason={reason}")
            return True

        except Exception as e:
            print(colored(f"❌ open error: {e}", "red"))
            logging.error(f"open_market error: {e}")
            return False
        finally:
            ENTRY_IN_PROGRESS = False
            PENDING_OPEN = False

def close_market_strict(reason="STRICT"):
    """
    إغلاق صارم آمن من المنصّة:
    - منع التوازي + نافذة حماية زمنية
    - cancel_all_orders قبل الغلق
    - limit IOC reduceOnly → market reduceOnly → market (fallback)
    - مصالحة بعد التنفيذ + حساب PnL مركّب + تسجيل زمن/شمعة الغلق
    """
    global compound_pnl, LAST_CLOSE_TS, CLOSE_IN_PROGRESS, _last_close_attempt_ts, LAST_CLOSE_BAR_TS

    if CLOSE_LOCK.locked() or CLOSE_IN_PROGRESS:
        print(colored("⏸️ close in progress — skip", "yellow")); return
    if _now() - _last_close_attempt_ts < CLOSE_GUARD_WINDOW_SEC:
        print(colored("⏸️ close guard window — skip", "yellow")); return

    with CLOSE_LOCK:
        CLOSE_IN_PROGRESS = True
        _last_close_attempt_ts = _now()
        try:
            exch_qty, exch_side, exch_entry = _read_position()
            if exch_qty <= 0:
                if STATE.get("open"):
                    _reset_after_close(reason, prev_side=STATE.get("side"))
                    LAST_CLOSE_TS = time.time()
                return

            _cancel_symbol_orders()

            side_to_close = "sell" if (exch_side=="long") else "buy"
            qty_to_close  = safe_qty(exch_qty)

            try:
                bid, ask = _best_bid_ask()
            except Exception:
                bid = ask = None
            ref = (ask if exch_side=="long" else bid) or price_now() or STATE.get("entry")
            band_px = _price_band(side_to_close, ref, MAX_SLIP_CLOSE_BPS)
            link = _order_link("CLS")

            # 1) limit IOC reduceOnly
            try:
                if MODE_LIVE and band_px:
                    params = _params_close(); params.update({"timeInForce":"IOC", "orderLinkId": link})
                    ex.create_order(SYMBOL,"limit",side_to_close,qty_to_close,band_px,params)
                else:
                    print(colored(f"[PAPER] limit-IOC reduceOnly {side_to_close} {qty_to_close} @ {fmt(band_px)}", "cyan"))
            except Exception as e1:
                print(colored(f"⚠️ limit IOC close err: {e1}", "yellow"))
                # 2) market reduceOnly
                try:
                    if MODE_LIVE:
                        params = _params_close(); params.update({"orderLinkId": link})
                        ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
                    else:
                        print(colored(f"[PAPER] market reduceOnly {side_to_close} {qty_to_close}", "cyan"))
                except Exception as e2:
                    # 3) fallback market بدون reduceOnly بعد الإلغاء
                    if _bybit_reduceonly_reject(e2):
                        print(colored("↪️ reduceOnly rejected — market w/o reduceOnly (safe after cancel)", "yellow"))
                        params = {"positionSide":"BOTH","reduceOnly":False,"positionIdx":0,"timeInForce":"IOC","orderLinkId":link}
                        if MODE_LIVE:
                            ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
                        else:
                            print(colored(f"[PAPER] market Fallback {side_to_close} {qty_to_close}", "cyan"))
                    else:
                        raise e2

            time.sleep(1.0)
            left_qty, _, _ = _read_position()
            if left_qty <= 0:
                px = price_now() or ref
                entry_px = STATE.get("entry") or exch_entry or px
                side = STATE.get("side") or exch_side
                qty  = exch_qty
                pnl  = (px - entry_px) * qty * (1 if side=="long" else -1)
                compound_pnl += pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                _reset_after_close(reason, prev_side=side); LAST_CLOSE_TS = time.time(); return

            # retries Market reduceOnly
            for _ in range(3):
                qty_to_close = safe_qty(left_qty)
                try:
                    if MODE_LIVE:
                        params = _params_close(); params.update({"timeInForce":"IOC", "orderLinkId": _order_link("CLS")})
                        ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
                    else:
                        print(colored(f"[PAPER] market retry reduceOnly {side_to_close} {qty_to_close}", "cyan"))
                except Exception as e:
                    print(colored(f"⚠️ market close retry err: {e}", "yellow"))
                time.sleep(0.8)
                left_qty, _, _ = _read_position()
                if left_qty <= 0:
                    px = price_now() or ref
                    entry_px = STATE.get("entry") or exch_entry or px
                    side = STATE.get("side") or exch_side
                    qty  = exch_qty
                    pnl  = (px - entry_px) * qty * (1 if side=="long" else -1)
                    compound_pnl += pnl
                    print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                    logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                    _reset_after_close(reason, prev_side=side); LAST_CLOSE_TS = time.time(); return

            print(colored(f"❌ STRICT CLOSE FAILED — residual pos still exists", "red"))
        except Exception as e:
            print(colored(f"❌ close error: {e}", "red"))
            logging.error(f"close_market_strict error: {e}")
        finally:
            CLOSE_IN_PROGRESS = False

def _reset_after_close(reason, prev_side=None):
    global LAST_CLOSE_BAR_TS, POST_CHOP_BLOCK_ACTIVE, POST_CHOP_BLOCK_UNTIL_BAR
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None,
        "hp_pct": 0.0, "strength": 0.0,
        "peak_adx": 0.0, "rsi_peak": 50.0, "rsi_trough": 50.0,
        "peak_price": 0.0, "trough_price": 0.0,
        "opp_rf_count": 0, "scm_line": "", "chop_flag": False
    })
    LAST_CLOSE_BAR_TS = LAST_DECISION_BAR_TS
    if reason.startswith("CHOP"):
        POST_CHOP_BLOCK_ACTIVE = True
        POST_CHOP_BLOCK_UNTIL_BAR = (LAST_DECISION_BAR_TS or 0) + POST_CHOP_WAIT_BARS
    logging.info(f"AFTER_CLOSE reason={reason} prev_side={prev_side}")

# =================== Chop / Accumulation Detector ===================
def _bb_width_pct(d: pd.DataFrame) -> float:
    if len(d) < BB_LEN+2: return 999.0
    c = d["close"].astype(float)
    m = c.rolling(BB_LEN).mean()
    sd = c.rolling(BB_LEN).std().replace(0,1e-12)
    upper = m + 2*sd
    lower = m - 2*sd
    bw = (upper.iloc[-2] - lower.iloc[-2])
    mid= max(m.iloc[-2], 1e-12)
    return float((bw / mid) * 100.0)

def _atr_pct_now_vs_median(df: pd.DataFrame) -> float:
    if len(df) < CHOP_LOOKBACK+5: return 999.0
    closes = df["close"].astype(float)
    highs  = df["high"].astype(float)
    lows   = df["low"].astype(float)
    tr = pd.concat([(highs-lows).abs(), (highs-closes.shift(1)).abs(), (lows-closes.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)
    atr_pct = (atr / closes.replace(0,1e-12))*100.0
    cur = float(atr_pct.iloc[-2])  # آخر شمعة مغلقة
    med = float(atr_pct.iloc[-(CHOP_LOOKBACK+1):-1].median())
    return cur / max(med, 1e-9)

def _range_width_bps(df: pd.DataFrame, bars: int) -> float:
    if len(df) < bars+2: return 999.0
    d = df.iloc[-(bars+1):-1]
    hi = float(d["high"].max()); lo = float(d["low"].min()); mid = (hi+lo)/2.0
    if mid <= 0: return 999.0
    return abs((hi-lo)/mid)*10000.0

def is_chop_zone(df: pd.DataFrame, ind: dict) -> bool:
    """ADX منخفض + ATR% مضغوط + بولنجر ضيق + نطاق ضيق."""
    adx = float(ind.get("adx") or 0.0)
    if adx > CHOP_ADX_MAX: 
        return False
    atr_frac = _atr_pct_now_vs_median(df)
    bb_pct   = _bb_width_pct(df)
    rng_bps  = _range_width_bps(df, CHOP_RANGE_BARS)
    return (atr_frac <= CHOP_ATR_PCT_FRACTION) and (bb_pct <= CHOP_BB_WIDTH_PCT_MAX) and (rng_bps <= CHOP_RANGE_BPS_MAX)

# =================== متابعة الترند/الصفقة ===================
def _update_trend_state(ind, info):
    if not STATE["open"]: return
    adx = float(ind.get("adx") or 0.0)
    rsi = float(ind.get("rsi") or 50.0)
    px  = float(info.get("price") or STATE.get("entry") or 0.0)
    if adx > (STATE.get("peak_adx") or adx): STATE["peak_adx"]=adx
    if rsi > (STATE.get("rsi_peak") or rsi): STATE["rsi_peak"]=rsi
    if rsi < (STATE.get("rsi_trough") or rsi): STATE["rsi_trough"]=rsi
    if STATE["side"]=="long":
        if px > (STATE.get("peak_price") or px): STATE["peak_price"]=px
    else:
        if px < (STATE.get("trough_price") or px): STATE["trough_price"]=px

def _near_price_bps(a,b):
    try: return abs((a-b)/b)*10000.0
    except Exception: return 0.0

def wick_or_bigcandle_harvest(df, ind, info):
    if not STATE["open"]: return False
    px = info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr = (px - entry)/entry*100*(1 if side=="long" else -1)
    if rr < WICK_TAKE_MIN_PCT: return False
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
    l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
    atr=float(ind.get("atr") or 0.0); body=abs(c-o)
    big_body = (atr>0 and body >= BODY_BIG_ATR_MULT*atr)
    big_wick_up   = (upper/rng)>=WICK_BIG_RATIO
    big_wick_down = (lower/rng)>=WICK_BIG_RATIO
    if side=="long" and (big_body or big_wick_up):  close_market_strict("WICK/BIGCANDLE_HARVEST"); return True
    if side=="short" and (big_body or big_wick_down): close_market_strict("WICK/BIGCANDLE_HARVEST"); return True
    return False

def _bos_against_trend(df: pd.DataFrame, side: str) -> bool:
    if len(df) < EXH_BOS_LOOKBACK+1: return False
    d = df.iloc[:-1]
    closes = d["close"].astype(float).values
    highs  = d["high"].astype(float).values
    lows   = d["low"].astype(float).values
    if side=="long":
        last_low = min(lows[-EXH_BOS_LOOKBACK:])
        return closes[-1] < last_low
    else:
        last_high = max(highs[-EXH_BOS_LOOKBACK:])
        return closes[-1] > last_high

def council_exhaustion_votes(df, ind, info, zones, trend):
    if len(df)<1 or not STATE["open"]: return 0, []
    side = STATE["side"]; reasons=[]; votes=0
    adx = float(ind.get("adx") or 0.0)
    rsi = float(ind.get("rsi") or 50.0)
    pdi = float(ind.get("plus_di") or 0.0); mdi = float(ind.get("minus_di") or 0.0)
    px  = float(info.get("price") or STATE["entry"]); entry = float(STATE["entry"])
    rr_pct = (px - entry)/entry*100.0*(1 if side=="long" else -1)
    if rr_pct < EXH_MIN_PNL_PCT: return 0, ["profit<threshold"]

    peak_adx = float(STATE.get("peak_adx") or adx)
    if peak_adx - adx >= EXH_ADX_DROP: votes += 1; reasons.append(f"ADX drop {peak_adx:.1f}->{adx:.1f}")
    if adx < EXH_ADX_MIN:              votes += 1; reasons.append(f"ADX<{EXH_ADX_MIN}")

    if side=="long" and mdi > pdi: votes += 1; reasons.append("DI- > DI+")
    if side=="short" and pdi > mdi: votes += 1; reasons.append("DI+ > DI-")

    if side=="long":
        if STATE.get("rsi_peak", rsi) - rsi >= EXH_RSI_PULLBACK and STATE.get("rsi_peak", rsi) >= 70:
            votes += 1; reasons.append("RSI retreat from OB")
    else:
        if rsi - STATE.get("rsi_trough", rsi) >= EXH_RSI_PULLBACK and STATE.get("rsi_trough", rsi) <= 30:
            votes += 1; reasons.append("RSI retreat from OS")

    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1]); l=float(df["low"].iloc[-1]); c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
    sup, dem = zones.get("supply"), zones.get("demand")
    if side=="long" and sup and (upper/rng)>=EXH_WICK_RATIO: votes += 1; reasons.append("upper wick near supply")
    if side=="short" and dem and (lower/rng)>=EXH_WICK_RATIO: votes += 1; reasons.append("lower wick near demand")

    hyst = _near_price_bps(info["price"], info["filter"])
    if side=="long" and info.get("short") and hyst>=EXH_HYST_MIN_BPS:
        if trend=="strong_up":
            if STATE.get("opp_rf_count",0) >= OPP_RF_DEBOUNCE: votes += 1; reasons.append("opp RF debounced")
        else: votes += 1; reasons.append("opp RF")
    if side=="short" and info.get("long") and hyst>=EXH_HYST_MIN_BPS:
        if trend=="strong_down":
            if STATE.get("opp_rf_count",0) >= OPP_RF_DEBOUNCE: votes += 1; reasons.append("opp RF debounced")
        else: votes += 1; reasons.append("opp RF")

    if _bos_against_trend(df, side): votes += 1; reasons.append("BOS against trend")
    return votes, reasons

def manage_position(df, ind, info, zones, trend):
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr = (px - entry)/entry*100*(1 if side=="long" else -1)

    opp = (side=="long" and info.get("short")) or (side=="short" and info.get("long"))
    STATE["opp_rf_count"] = STATE.get("opp_rf_count",0)+1 if opp else 0

    if wick_or_bigcandle_harvest(df, ind, info): return

    # NEW: Chop Exit — لو منطقة تذبذب/تجميع و ربح بسيط → غلق صارم
    choppy = is_chop_zone(df, ind)
    STATE["chop_flag"] = bool(choppy)
    if choppy and rr >= CHOP_MIN_PNL_PCT and trend not in ("strong_up","strong_down"):
        close_market_strict("CHOP_EXIT"); return

    votes, rs = council_exhaustion_votes(df, ind, info, zones, trend)
    if votes >= EXH_VOTES_NEEDED:
        close_market_strict("SCM_EXHAUSTION: " + ",".join(rs)); return

    xp = xprotect_signal(df, ind, info)
    if side=="long" and xp["explode_down"]:
        close_market_strict("XPROTECT_LONG_EXPLODE_DOWN"); return
    if side=="short" and xp["explode_up"]:
        close_market_strict("XPROTECT_SHORT_EXPLODE_UP"); return

    adx=float(ind.get("adx") or 0.0)
    hyst=_near_price_bps(info["price"], info["filter"])
    if opp and adx>=BREAK_ADX_MIN and hyst>=EXH_HYST_MIN_BPS:
        if (trend=="strong_up" and side=="long" and STATE["opp_rf_count"]>=OPP_RF_DEBOUNCE) or \
           (trend=="strong_down" and side=="short" and STATE["opp_rf_count"]>=OPP_RF_DEBOUNCE) or \
           (trend not in ("strong_up","strong_down")):
            close_market_strict("OPPOSITE_RF_CONFIRMED"); return

    if rr >= TRAIL_ACTIVATE_PCT and ind.get("atr",0)>0:
        gap = ind["atr"] * ATR_TRAIL_MULT
        if side=="long":
            new_trail = px - gap
            STATE["trail"] = max(STATE["trail"] or new_trail, new_trail)
            if px < STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)"); return
        else:
            new_trail = px + gap
            STATE["trail"] = min(STATE["trail"] or new_trail, new_trail)
            if px > STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)"); return

# =================== مصالحة الحالة مع المنصّة ===================
def reconcile_state():
    exch_qty, exch_side, exch_entry = _read_position()
    if (exch_qty or 0.0) <= 0:
        if STATE.get("open"):
            print(colored("🧹 RECONCILE: exchange flat, resetting local state.", "yellow"))
            _reset_after_close("RECONCILE_FLAT", prev_side=STATE.get("side"))
        return
    changed = (not STATE.get("open")) or \
              (STATE.get("side") != exch_side) or \
              (abs((STATE.get("qty") or 0) - exch_qty) > (LOT_STEP or 0.0)) or \
              (abs((STATE.get("entry") or 0) - exch_entry) / max(exch_entry,1e-9) > 0.001)
    if changed:
        STATE.update({
            "open": True, "side": exch_side, "entry": float(exch_entry),
            "qty": safe_qty(exch_qty)
        })
        print(colored(f"🔄 RECONCILE: synced to exchange — {exch_side} qty={fmt(exch_qty,4)} @ {fmt(exch_entry)}", "cyan"))

# =================== Snapshot ===================
def _last_closed_bar_ts(df):
    if len(df) >= 2: return int(df["time"].iloc[-2])
    return int(df["time"].iloc[-1]) if len(df) else 0

def pretty_snapshot(bal, info, ind, spread_bps, zones, reason=None, df=None):
    left_s = time_to_candle_close(df) if df is not None else 0
    print(colored("─"*110,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
    print("📈 RF & INDICATORS (CLOSED)")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))} hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)} bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
    print(f"   🏗️ ZONES: {zones}")
    print(f"   🧠 {STATE.get('scm_line','')}")
    print(f"   🧊 CHOP={STATE.get('chop_flag', False)}  | POST_CHOP_BLOCK={POST_CHOP_BLOCK_ACTIVE}")
    print(f"   ⏱️ closes_in ≈ {left_s}s")

    print("\n🧭 POSITION")
    bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(colored(f"   {bal_line}", "yellow"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  HP={fmt(STATE['hp_pct'],2)}%  OppRF={STATE.get('opp_rf_count',0)}  Strength={fmt(STATE.get('strength',0),2)}")
    else:
        print("   ⚪ FLAT")
    if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
    print(colored("─"*110,"cyan"))

# =================== EVALUATE ===================
def evaluate_all(df):
    info = rf_signal_closed(df)
    ind  = compute_indicators(df)
    zones = detect_zones(df)
    candidates, trend = council_entry(df, ind, info, zones)
    return info, ind, zones, candidates, trend

# =================== LOOP ===================
def trade_loop():
    global LAST_CLOSE_TS, LAST_DECISION_BAR_TS, _last_entry_attempt_ts
    global POST_CHOP_BLOCK_ACTIVE, POST_CHOP_BLOCK_UNTIL_BAR, LAST_CLOSE_BAR_TS
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()

            # مصالحة مستمرة
            reconcile_state()

            info, ind, zones, candidates, trend = evaluate_all(df)

            spread_bps = orderbook_spread_bps()
            reason=None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"

            if reason is None and (time.time() - LAST_CLOSE_TS) < COOLDOWN_SEC:
                reason = f"cooldown {(COOLDOWN_SEC - int(time.time()-LAST_CLOSE_TS))}s"
            while TRADE_TIMES and time.time()-TRADE_TIMES[0] > 3600:
                TRADE_TIMES.popleft()
            if reason is None and len(TRADE_TIMES) >= MAX_TRADES_PER_HOUR:
                reason = "rate-limit: too many trades this hour"

            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
                STATE["hp_pct"] = max(STATE.get("hp_pct",0.0), (px-STATE["entry"])/STATE["entry"]*100.0*(1 if STATE["side"]=="long" else -1))
                _update_trend_state(ind, {"price":px, **info})

            # إدارة الصفقة
            manage_position(df, ind, {"price": px or info["price"], **info}, zones, trend)

            # قرار واحد لكل شمعة مغلقة
            bar_ts = _last_closed_bar_ts(df)
            decision_allowed = (bar_ts != LAST_DECISION_BAR_TS)

            # رفع بلوك ما بعد التذبذب بعد عدد شموع
            if POST_CHOP_BLOCK_ACTIVE and bar_ts >= POST_CHOP_BLOCK_UNTIL_BAR:
                # لو مطلوب RF بعد التذبذب ولم تظهر إشارة RF مغلقة، أبقِ البلوك
                if POST_CHOP_REQUIRE_RF and not (info.get("long") or info.get("short")):
                    pass
                else:
                    POST_CHOP_BLOCK_ACTIVE = False

            best = None
            if decision_allowed and not STATE["open"] and reason is None:
                # منع إعادة الدخول مباشرة بعد الغلق (شمعة واحدة كحد أدنى)
                if LAST_CLOSE_BAR_TS and bar_ts <= LAST_CLOSE_BAR_TS + MIN_REENTRY_BARS - 1:
                    reason = "min reentry bars guard"
                else:
                    best = choose_best_entry(candidates, ind)
                    # Gate ضد انفجار عكسي
                    xp_gate = xprotect_signal(df, ind, {"price": px or info["price"], **info})
                    if best and ((best["side"]=="buy"  and xp_gate["explode_down"]) or
                                 (best["side"]=="sell" and xp_gate["explode_up"])):
                        reason = f"gate: xprotect against entry ({xp_gate['why']})"
                        best = None

            if decision_allowed and not STATE["open"] and best and reason is None:
                adx_now = float(ind.get("adx") or 0.0)
                if _now() - _last_entry_attempt_ts < ENTRY_GUARD_WINDOW_SEC:
                    reason = "entry guard window"
                else:
                    if best["src"] == "rf":
                        if adx_now < ADX_ENTRY_MIN:
                            reason = f"ignored RF — ADX<{ADX_ENTRY_MIN}"
                        else:
                            qty = compute_size(bal, px or info["price"])
                            ok = open_market("buy" if best["side"]=="buy" else "sell",
                                             qty, px or info["price"], best["score"], best["reason"])
                            _last_entry_attempt_ts = _now()
                            if not ok: reason="open failed (rf)"
                    else:  # council
                        if adx_now < BREAK_ADX_MIN or best.get("votes",0) < COUNCIL_ENTRY_VOTES_MIN or best["score"] < COUNCIL_STRONG_SCORE_MIN:
                            reason = "ignored Council — weak confirmation"
                        else:
                            qty = compute_size(bal, px or info["price"])
                            ok = open_market("buy" if best["side"]=="buy" else "sell",
                                             qty, px or info["price"], best["score"], "Council Strong Consensus")
                            _last_entry_attempt_ts = _now()
                            if not ok: reason="open failed (council)"
                LAST_DECISION_BAR_TS = bar_ts
            elif decision_allowed:
                LAST_DECISION_BAR_TS = bar_ts

            pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, zones, reason, df)

            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"] += 1

            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP
            time.sleep(sleep_s)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# =================== API / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ BYBIT SUI BOT — {SYMBOL} {INTERVAL} — {mode} — RF Closed + Strong Council • SmartExec + ChopExit"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS,
                   "post_chop_block": POST_CHOP_BLOCK_ACTIVE}
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "tp_done": STATE.get("hp_pct",0.0), "opp_votes": STATE.get("opp_rf_count",0),
        "chop": STATE.get("chop_flag", False), "post_chop_block": POST_CHOP_BLOCK_ACTIVE
    }), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (SELF_URL not set)", "yellow"))
        return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"bybit-sui-keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}", "cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  RF=Closed-candle", "yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading as _t
    _t.Thread(target=trade_loop, daemon=True).start()
    _t.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
