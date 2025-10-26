# -*- coding: utf-8 -*-
"""
BYBIT — SOL Perp Bot
RF (closed candle) + Smart Council SCM (Boxes + Liquidity + Sweeps + Displacement + Retest + Trap + Trend)
• ENV فقط: BYBIT_API_KEY, BYBIT_API_SECRET, SELF_URL/RENDER_EXTERNAL_URL, PORT
• دخول واحد مُحكّم (Council > RF). لا تكرار.
• Debounce لإشارات RF المعاكسة داخل الترند.
• Wick-harvest مؤجل في الترند القوي.
• خروج صارم عند نهاية موجة الترند بعد تأكيد SCM.
• Logs سطر واحد: «SCM | trend | boxes | liquidity | displacement | retest | trap | votes».
"""

import os, time, math, random, signal, sys, traceback, logging
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

# =================== ENV (المطلوب فقط) ===================
API_KEY  = os.getenv("BYBIT_API_KEY", "")
API_SEC  = os.getenv("BYBIT_API_SECRET", "")
SELF_URL = (os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")).strip()
PORT     = int(os.getenv("PORT", "5000"))

# =================== ثوابت رئيسية ===================
SYMBOL        = "SOL/USDT:USDT"
INTERVAL      = "15m"

LEVERAGE      = 10
RISK_ALLOC    = 0.60
POSITION_MODE = "oneway"

# RF (على الشمعة المُغلقة)
RF_SOURCE   = "close"
RF_PERIOD   = 20
RF_MULT     = 3.5
RF_HYST_BPS = 6.0

# مؤشرات
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# حُرّاس الدخول
ADX_ENTRY_MIN   = 17.0
MAX_SPREAD_BPS  = 8.0
COOLDOWN_SEC    = 90
MAX_TRADES_PER_HOUR = 6

# مناطق/صناديق
LEVEL_NEAR_BPS  = 12.0
WICK_RATIO_MIN  = 0.55

# SCM — سيولة / اختراق / إعادة اختبار / فِخاخ / ترند
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
WICK_TAKE_MIN_PCT  = 0.40

# خروج نهاية موجة (SCM Exhaustion)
EXH_MIN_PNL_PCT   = 0.35
EXH_ADX_DROP      = 6.0
EXH_ADX_MIN       = 18.0
EXH_RSI_PULLBACK  = 7.0
EXH_WICK_RATIO    = 0.60
EXH_HYST_MIN_BPS  = 8.0
EXH_BOS_LOOKBACK  = 6
EXH_VOTES_NEEDED  = 3

# تصويت الدخول
COUNCIL_ENTRY_VOTES_MIN  = 4
COUNCIL_STRONG_SCORE_MIN = 3.5

# إيقاع
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# =================== لوج ===================
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

# =================== البورصة ===================
def make_ex():
    return ccxt.bybit({
        "apiKey": API_KEY,
        "secret": API_SEC,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"}
    })
ex = make_ex()
MODE_LIVE = bool(API_KEY and API_SEC)
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
    "scm_line": ""
}

def _norm_sym(s: str) -> str:
    return (s or "").replace("/", "").replace(":", "").upper()
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

# =================== مؤشرات / RF ===================
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
    d = df.iloc[:-1].copy()
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

def _near_bps(px, lvl, bps):
    try: return abs((px-lvl)/lvl)*10000.0 <= bps
    except Exception: return False

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

# =================== SCM Signals ===================
def displacement_bar(df: pd.DataFrame, ind: dict, side: str) -> bool:
    if len(df) < 1: return False
    o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
    atr=float(ind.get("atr") or 0.0); body=abs(c-o)
    if atr<=0: return False
    if side=="buy":  return (c>o) and (body >= DISP_BODY_ATR_MIN*atr)
    else:            return (c<o) and (body >= DISP_BODY_ATR_MIN*atr)

def retest_happened(history_df: pd.DataFrame, zones: dict, side: str) -> bool:
    """إعادة اختبار سريعة لصندوق مكسور خلال RETEST_MAX_BARS شموع مغلقة."""
    try:
        if len(history_df) < RETEST_MAX_BARS + 2:
            return False
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

def trap_detect(df: pd.DataFrame, zones: dict, side: str) -> bool:
    if len(df) < 2: return False
    c=float(df["close"].iloc[-1]); o=float(df["open"].iloc[-1])
    sup, dem = zones.get("supply"), zones.get("demand")
    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    if side=="buy" and sup and c<sup["top"] and _bps(o, sup["top"])>=BREAK_HYST_BPS and _bps(c, sup["top"])<=TRAP_CLOSE_BACK_BPS:
        return True
    if side=="sell" and dem and c>dem["bot"] and _bps(o, dem["bot"])>=BREAK_HYST_BPS and _bps(c, dem["bot"])<=TRAP_CLOSE_BACK_BPS:
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

# =================== Votes (Council SCM) ===================
def council_scm_votes(df, ind, info, zones):
    reasons_b=[]; reasons_s=[]; b=s=0; score_b=0.0; score_s=0.0
    trend = trend_context(ind)

    # Boxes (breakout)
    o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
    sup, dem = zones.get("supply"), zones.get("demand")
    atr=float(ind.get("atr") or 0.0); adx=float(ind.get("adx") or 0.0)
    pdi=float(ind.get("plus_di") or 0.0); mdi=float(ind.get("minus_di") or 0.0)
    body=abs(c-o)
    def _bps(a,b): 
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0

    boxes="—"
    if sup and c>sup["top"] and _bps(c,sup["top"])>=BREAK_HYST_BPS and body>=BREAK_BODY_ATR_MIN*max(atr,1e-9) and adx>=BREAK_ADX_MIN and (pdi>=mdi+BREAK_DI_MARGIN):
        b+=2; score_b+=1.6; reasons_b.append("breakout@supply +2"); boxes="↑sup"
    if dem and c<dem["bot"] and _bps(c,dem["bot"])>=BREAK_HYST_BPS and body>=BREAK_BODY_ATR_MIN*max(atr,1e-9) and adx>=BREAK_ADX_MIN and (mdi>=pdi+BREAK_DI_MARGIN):
        s+=2; score_s+=1.6; reasons_s.append("breakout@demand +2"); boxes="↓dem" if boxes=="—" else boxes+"|↓dem"

    # Liquidity (EQH/EQL + Sweeps)
    eqh,eql = find_equal_highs_lows(df)
    sw = detect_sweep(df, eqh, eql)
    liquidity="—"
    if sw["sweep_up"]:
        s+=1; score_s+=0.6; reasons_s.append("sweep_up"); liquidity="sweep_up"
    if sw["sweep_down"]:
        b+=1; score_b+=0.6; reasons_b.append("sweep_down"); liquidity="sweep_down" if liquidity=="—" else liquidity+"|sweep_down"
    if eqh: liquidity = ("EQH" if liquidity=="—" else liquidity+"|EQH")
    if eql: liquidity = ("EQL" if liquidity=="—" else liquidity+"|EQL")

    # Displacement
    disp_b = displacement_bar(df, ind, "buy")
    disp_s = displacement_bar(df, ind, "sell")
    displacement="—"
    if disp_b: b+=1; score_b+=0.7; reasons_b.append("displacement+"); displacement="up"
    if disp_s: s+=1; score_s+=0.7; reasons_s.append("displacement-"); displacement="down" if displacement=="—" else displacement+"|down"

    # Retest
    ret_b = retest_happened(df, zones, "buy")
    ret_s = retest_happened(df, zones, "sell")
    retest="—"
    if ret_b: b+=1; score_b+=0.5; reasons_b.append("retest@supbreak"); retest="up"
    if ret_s: s+=1; score_s+=0.5; reasons_s.append("retest@dembreak"); retest="down" if retest=="—" else retest+"|down"

    # Trap
    trap_b = trap_detect(df, zones, "buy")
    trap_s = trap_detect(df, zones, "sell")
    trap="—"
    if trap_b: b+=1; score_b+=0.6; reasons_b.append("trap@sup"); trap="bull"
    if trap_s: s+=1; score_s+=0.6; reasons_s.append("trap@dem"); trap=("bear" if trap=="—" else trap+"|bear")

    # RF كمساعدة
    if info.get("long"):  b+=1; score_b+=0.5; reasons_b.append("rf_long")
    if info.get("short"): s+=1; score_s+=0.5; reasons_s.append("rf_short")

    # اتجاه DI/ADX
    if pdi>mdi and adx>=18: b+=1; score_b+=0.5; reasons_b.append("DI+>DI- & ADX")
    if mdi>pdi and adx>=18: s+=1; score_s+=0.5; reasons_s.append("DI->DI+ & ADX")

    score_b += b/4.0; score_s += s/4.0
    scm_line = f"SCM | {trend} | {boxes} | {liquidity} | {displacement} | {retest} | {trap} | votes(b={b},s={s})"
    return b,reasons_b,s,reasons_s,score_b,score_s,scm_line,trend

# =================== Council Exhaustion (خروج) ===================
def _near_price_bps(a,b):
    try: return abs((a-b)/b)*10000.0
    except Exception: return 0.0

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

# =================== Council ENTRY Arbitration ===================
def council_entry(df, ind, info, zones):
    b,b_r,s,s_r,score_b,score_s,scm_line,trend = council_scm_votes(df, ind, info, zones)
    STATE["scm_line"] = scm_line
    candidates=[]
    if b >= COUNCIL_ENTRY_VOTES_MIN:
        candidates.append({"side":"buy","score":score_b,"reason":f"SCM BUY {b} :: {b_r}", "trend":trend})
    if s >= COUNCIL_ENTRY_VOTES_MIN:
        candidates.append({"side":"sell","score":score_s,"reason":f"SCM SELL {s} :: {s_r}", "trend":trend})
    if info.get("long"):  candidates.append({"side":"buy","score":1.0,"reason":"RF_LONG", "trend":trend})
    if info.get("short"): candidates.append({"side":"sell","score":1.0,"reason":"RF_SHORT","trend":trend})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates, trend

def choose_best_entry(candidates, ind):
    if not candidates: return None
    best = candidates[0]
    pdi=float(ind.get("plus_di") or 0.0); mdi=float(ind.get("minus_di") or 0.0)
    if len(candidates)>=2 and candidates[0]["score"]==candidates[1]["score"]:
        if best["side"]=="buy" and pdi<mdi: best = candidates[1]
        if best["side"]=="sell" and mdi<pdi: best = candidates[1]
    return best

# =================== أوامر ===================
def _params_open(side):
    if POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if side=="buy" else "SHORT", "reduceOnly": False}
    return {"positionSide": "BOTH", "reduceOnly": False}
def _params_close():
    if POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if STATE.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide": "BOTH", "reduceOnly": True}

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

# --------- حجم آمن + فتح تكيفي (مُصحَّح نهائي) ---------
def compute_size(balance, price):
    """
    حجم آمن قابل للتنفيذ:
    - Buffer للرسوم/الانزلاق (3%).
    - يلتزم بقيود اللوت والمينيمم.
    """
    equity = float(balance or 0.0)
    px = max(float(price or 0.0), 1e-9)
    buffer = 0.97  # 3% أمان
    notional = equity * RISK_ALLOC * LEVERAGE * buffer
    raw_qty = notional / px
    return safe_qty(raw_qty)

def open_market(side, qty, price, strength, reason):
    """
    فتح تكيفي:
    1) كاب بالحجم الأقصى المناسب للرصيد.
    2) عند خطأ الهامش (-110007 / "not enough"/"insufficient") نقلّص 15% ونُعيد (حتى 6 مرات).
    3) احترام LOT_MIN/LOT_STEP دائماً.
    """
    bal = balance_usdt()
    px = float(price or 0.0)
    max_affordable = compute_size(bal, px)
    q_try = safe_qty(min(qty, max_affordable))

    if q_try <= 0 or (LOT_MIN and q_try < LOT_MIN):
        print(colored(f"❌ skip open (qty too small) — bal={fmt(bal,2)} px={fmt(px)} q={fmt(q_try,4)}", "red"))
        return False

    attempts = 0
    while attempts < 6:
        try:
            if MODE_LIVE:
                try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
                except Exception: pass
                ex.create_order(SYMBOL, "market", side, q_try, None, _params_open(side))
            # success
            STATE.update({
                "open": True, "side": "long" if side=="buy" else "short", "entry": price,
                "qty": q_try, "pnl": 0.0, "bars": 0, "trail": None,
                "hp_pct": 0.0, "strength": float(strength),
                "peak_adx": 0.0, "rsi_peak": 50.0, "rsi_trough": 50.0,
                "peak_price": float(price), "trough_price": float(price),
                "opp_rf_count": 0
            })
            TRADE_TIMES.append(time.time())
            print(colored(
                f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} "
                f"qty={fmt(q_try,4)} @ {fmt(price)} | strength={fmt(strength,2)} | {reason}",
                "green" if side=='buy' else 'red'
            ))
            logging.info(f"OPEN {side} qty={q_try} price={price} strength={strength} reason={reason}")
            return True

        except Exception as e:
            msg = str(e).lower()
            attempts += 1
            insufficient = ("-110007" in msg) or ("not enough" in msg) or ("insufficient" in msg)
            if insufficient:
                new_q = safe_qty(q_try * 0.85)
                print(colored(f"⚠️ open: insufficient margin — retry {attempts}/6 with qty={fmt(new_q,4)} (was {fmt(q_try,4)})", "yellow"))
                if new_q <= 0 or (LOT_MIN and new_q < LOT_MIN):
                    print(colored("⛔ qty fell below exchange minimum — abort open", "red"))
                    break
                q_try = new_q
                time.sleep(0.5)
                continue
            print(colored(f"❌ open: {e}", "red")); logging.error(f"open_market error: {e}"); break
    return False
# --------------------------------------------------------

def _reset_after_close(reason, prev_side=None):
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None,
        "hp_pct": 0.0, "strength": 0.0,
        "peak_adx": 0.0, "rsi_peak": 50.0, "rsi_trough": 50.0,
        "peak_price": 0.0, "trough_price": 0.0,
        "opp_rf_count": 0,
        "scm_line": ""
    })
    logging.info(f"AFTER_CLOSE reason={reason} prev_side={prev_side}")

def close_market_strict(reason="STRICT"):
    global compound_pnl, LAST_CLOSE_TS
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0:
        if STATE.get("open"):
            _reset_after_close(reason, prev_side=STATE.get("side"))
            LAST_CLOSE_TS = time.time()
        return
    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts=0; last_error=None
    while attempts < 6:
        try:
            if MODE_LIVE:
                params = _params_close(); params["reduceOnly"]=True
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(1.5)
            left_qty, _, _ = _read_position()
            if left_qty <= 0:
                px = price_now() or STATE.get("entry")
                entry_px = STATE.get("entry") or exch_entry or px
                side = STATE.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
                qty  = exch_qty
                pnl  = (px - entry_px) * qty * (1 if side=="long" else -1)
                compound_pnl += pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                _reset_after_close(reason, prev_side=side); LAST_CLOSE_TS = time.time(); return
            qty_to_close = safe_qty(left_qty); attempts += 1
            print(colored(f"⚠️ strict close retry {attempts}/6 — residual={fmt(left_qty,4)}","yellow"))
            time.sleep(1.5)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(1.5)
    print(colored(f"❌ STRICT CLOSE FAILED — last error: {last_error}", "red"))

# =================== إدارة الصفقة ===================
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

def manage_position(df, ind, info, zones, trend):
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr = (px - entry)/entry*100*(1 if side=="long" else -1)

    opp = (side=="long" and info.get("short")) or (side=="short" and info.get("long"))
    STATE["opp_rf_count"] = STATE.get("opp_rf_count",0)+1 if opp else 0

    votes, rs = council_exhaustion_votes(df, ind, info, zones, trend)
    if votes >= EXH_VOTES_NEEDED:
        close_market_strict("SCM_EXHAUSTION: " + ",".join(rs)); return

    adx=float(ind.get("adx") or 0.0)
    hyst=_near_price_bps(info["price"], info["filter"])
    if opp and adx>=BREAK_ADX_MIN and hyst>=EXH_HYST_MIN_BPS:
        if (trend=="strong_up" and side=="long" and STATE["opp_rf_count"]>=OPP_RF_DEBOUNCE) or \
           (trend=="strong_down" and side=="short" and STATE["opp_rf_count"]>=OPP_RF_DEBOUNCE) or \
           (trend not in ("strong_up","strong_down")):
            close_market_strict("OPPOSITE_RF_CONFIRMED"); return

    if rr >= WICK_TAKE_MIN_PCT and trend not in ("strong_up","strong_down"):
        o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1]); l=float(df["low"].iloc[-1]); c=float(df["close"].iloc[-1])
        rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
        if (side=="long" and (upper/rng)>=WICK_RATIO_MIN) or (side=="short" and (lower/rng)>=WICK_RATIO_MIN):
            close_market_strict("LONG_WICK_TAKE"); return

    atr=float(ind.get("atr") or 0.0)
    if rr >= TRAIL_ACTIVATE_PCT and atr>0:
        gap = atr * ATR_TRAIL_MULT
        if side=="long":
            new_trail = px - gap
            STATE["trail"] = max(STATE["trail"] or new_trail, new_trail)
            if px < STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")
        else:
            new_trail = px + gap
            STATE["trail"] = min(STATE["trail"] or new_trail, new_trail)
            if px > STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")
    if rr > STATE["hp_pct"]: STATE["hp_pct"]=rr

# =================== Snapshot ===================
def pretty_snapshot(bal, info, ind, spread_bps, zones, reason=None, df=None, cand=None):
    left_s = time_to_candle_close(df) if df is not None else 0
    print(colored("─"*110,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
    print(f"SCM: {STATE.get('scm_line','—')}")
    print("📈 RF & INDICATORS (CLOSED)")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))} hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)} bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
    print(f"   🏗️ ZONES: {zones}")
    print(f"   ⏱️ closes_in ≈ {left_s}s")
    if cand: print(colored(f"   🎛️ candidate ⇒ {cand}", "white"))
    print("\n🧭 POSITION")
    bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(colored(f"   {bal_line}", "yellow"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  HP={fmt(STATE['hp_pct'],2)}%  Strength={fmt(STATE['strength'],2)}  OppRF={STATE.get('opp_rf_count',0)}")
    else:
        print("   ⚪ FLAT")
    if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
    print(colored("─"*110,"cyan"))

# =================== Loop ===================
def evaluate_all(df):
    info = rf_signal_closed(df)
    ind  = compute_indicators(df)
    zones = detect_zones(df)
    candidates, trend = council_entry(df, ind, info, zones)
    return info, ind, zones, candidates, trend

def choose_and_open(best, bal, px, info):
    pre_qty = compute_size(bal, px or info["price"])
    return open_market("buy" if best["side"]=="buy" else "sell",
                       pre_qty, px or info["price"],
                       best["score"], best["reason"])

def trade_loop():
    global LAST_CLOSE_TS
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()

            info, ind, zones, candidates, trend = evaluate_all(df)
            best = choose_best_entry(candidates, ind)

            # Guards
            spread_bps = orderbook_spread_bps()
            reason=None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS: reason=f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            if not STATE["open"] and reason is None and float(ind.get("adx") or 0.0) < ADX_ENTRY_MIN: reason = f"ADX<{int(ADX_ENTRY_MIN)}: pause entries"
            if reason is None and (time.time() - LAST_CLOSE_TS) < COOLDOWN_SEC: reason = f"cooldown {(COOLDOWN_SEC - int(time.time()-LAST_CLOSE_TS))}s"
            while TRADE_TIMES and time.time()-TRADE_TIMES[0] > 3600: TRADE_TIMES.popleft()
            if reason is None and len(TRADE_TIMES) >= MAX_TRADES_PER_HOUR: reason = "rate-limit: too many trades this hour"

            # تحديث PnL وتتبع الترند
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            _update_trend_state(ind, {"price": px or info["price"]})

            # إدارة الصفقة (SCM + Debounce)
            manage_position(df, ind, {"price": px or info["price"], **info}, zones, trend)

            # انعكاس لقرار أقوى
            if STATE["open"] and best and reason is None:
                opposite = (best["side"]=="buy" and STATE["side"]=="short") or (best["side"]=="sell" and STATE["side"]=="long")
                margin = 0.5
                if opposite and best["score"] >= max(STATE.get("strength",0.0)+margin, COUNCIL_STRONG_SCORE_MIN):
                    close_market_strict("REVERSE_TO_STRONGER_DECISION")
                    bal = balance_usdt(); px = price_now()
                    choose_and_open(best, bal, px, info)

            # دخول إن فاضي
            if (not STATE["open"]) and best and reason is None:
                logging.info(f"ENTRY_CHOSEN {best} | {STATE.get('scm_line','')}")
                print(colored(f"ENTRY_CHOSEN {best}", "cyan"))
                if not choose_and_open(best, bal, px, info):
                    reason = "qty<=0 or open failed"

            pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, zones, reason, df, best)

            # عداد شموع
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"] += 1

            time.sleep(NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# =================== API / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ BYBIT SOL BOT — {SYMBOL} {INTERVAL} — {mode} — RF CLOSED + SCM Council + Debounce + Exhaustion Exit"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_gate": {"adx_min": ADX_ENTRY_MIN, "max_spread_bps": MAX_SPREAD_BPS},
        "scm_line": STATE.get("scm_line","")
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "hp_pct": STATE.get("hp_pct", 0.0), "strength": STATE.get("strength",0.0),
        "scm_line": STATE.get("scm_line","")
    }), 200

def keepalive_loop():
    url=SELF_URL.strip().rstrip("/")
    if not url:
        print(colored("⏸️ keepalive disabled (SELF_URL not set)", "yellow")); return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"bybit-sol-keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}", "cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored("SCM Council ON | Debounce RF | Exhaustion Exit | Adaptive Open + Safe Sizing", "yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
