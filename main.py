# -*- coding: utf-8 -*-
"""
BYBIT — SOL Perp Bot (RF-CLOSED + Smart Council • Single-Entry Arbitration)
• ENV فقط للمفاتيح والـ URL والبورت:
  - BYBIT_API_KEY, BYBIT_API_SECRET, SELF_URL (أو RENDER_EXTERNAL_URL), PORT
• الدخول على الشمعة المُغلقة فقط.
• مجلس الإدارة:
    - اختراقات صناديق Supply/Demand القوية (تصويت مضاعف + Score عالي)
    - مناطق شراء/بيع قوية (لمسة+فتيلة+تشبّع) حتى بدون اختراق
    - دمج DI/ADX/RSI + رفض مناطق + نماذج شموع
• تحكيم دخول واحد فقط (Council vs RF) حسب Score الأعلى — لا تكرار.
• انعكاس قوي: إغلاق صارم ثم فتح الصفقة الأقوى بالعكس.
• إدارة صفقة: ATR-Trail ديناميكي + Apex/Wick take + Opposite RF Confirmed
• Guards: ADX≥17, Spread, Cooldown, Rate-limit
• Flask /metrics /health + Rotating logs
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

# ============== ENV (المطلوب فقط) ==============
API_KEY  = os.getenv("BYBIT_API_KEY", "")
API_SEC  = os.getenv("BYBIT_API_SECRET", "")
SELF_URL = (os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")).strip()
PORT     = int(os.getenv("PORT", "5000"))

# =================== FIXED SETTINGS ===================
SYMBOL        = "SOL/USDT:USDT"
INTERVAL      = "15m"

LEVERAGE      = 10
RISK_ALLOC    = 0.60
POSITION_MODE = "oneway"   # "oneway" or "hedge"

# Range Filter (CLOSED candle only)
RF_SOURCE     = "close"
RF_PERIOD     = 20
RF_MULT       = 3.5
RF_HYST_BPS   = 6.0

# Indicators
RSI_LEN       = 14
ADX_LEN       = 14
ATR_LEN       = 14

# Entry gates / guards
ADX_ENTRY_MIN = 17.0
MAX_SPREAD_BPS = 8.0
COOLDOWN_SEC   = 90
MAX_TRADES_PER_HOUR = 6

# Council: zones & votes
LEVEL_NEAR_BPS           = 12.0
RSI_NEUTRAL_MIN          = 45.0
RSI_NEUTRAL_MAX          = 55.0
COUNCIL_ENTRY_VOTES_MIN  = 4            # الحد الأدنى للدخول
COUNCIL_STRONG_SCORE_MIN = 3.5          # اعتباره قرار قوي

# Breakout logic (Council)
BREAK_HYST_BPS      = 10.0  # مقدار الاختراق فوق/تحت الصندوق بالـbps
BREAK_ADX_MIN       = 22.0
BREAK_DI_MARGIN     = 5.0   # تفوق DI باتجاه الاختراق
BREAK_BODY_ATR_MIN  = 0.60  # جسم الشمعة ≥60% ATR

# Counter-trend (اختياري) — منطقة بيع/شراء قوية بدون اختراق
COUNTER_ENABLE          = True
CT_ADX_MAX              = 28.0  # تجنب مضاد ترند لو الترند قوي جدًا
SUPPLY_TOUCH_BPS        = 12.0
DEMAND_TOUCH_BPS        = 12.0
WICK_RATIO_MIN          = 0.55
CT_HYST_MIN_BPS         = 5.0

# One-shot management / exits
WICK_TAKE_MIN_PCT   = 0.40
TRAIL_ACTIVATE_PCT  = 1.20
ATR_TRAIL_MULT      = 1.6
OPP_RF_MIN_ADX      = 22.0
OPP_RF_MIN_HYST_BPS = 8.0

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# =================== LOGGING ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    exists = False
    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler) and getattr(h,"baseFilename","").endswith("bot.log"):
            exists = True
            break
    if not exists:
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready", "cyan"))

setup_file_logging()

# =================== EXCHANGE ===================
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

# =================== HELPERS / STATE ===================
LAST_CLOSE_TS = 0
TRADE_TIMES = deque(maxlen=10)
compound_pnl = 0.0

STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None,
    "hp_pct": 0.0, "strength": 0.0  # strength = score للقرار اللي دخلنا بيه
}

def _norm_sym(s: str) -> str:
    return (s or "").replace("/", "").replace(":", "").upper()

def _sym_match(a: str, b: str) -> bool:
    A, B = _norm_sym(a), _norm_sym(b)
    return A == B or A in B or B in A

def _round_amt(q):
    """Robust amount rounding including precision=0 + fractional LOT_MIN."""
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = AMT_PREC
        if (not prec or prec<=0) and LOT_MIN and LOT_MIN < 1:
            try:
                prec = max(1, -Decimal(str(LOT_MIN)).as_tuple().exponent)
            except Exception:
                prec = 1
        d = d.quantize(Decimal(1).scaleb(-int(prec or 0)), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)):
            return 0.0
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
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    return int(max(0, next_close_ms - now_ms)/1000)

# =================== INDICATORS / RF ===================
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
    """RF على الشمعة المُغلقة: يعتبر حالة فوق/تحت الفلتر بهستريسس، لا يشترط flip."""
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

    return {
        "time": int(d["time"].iloc[-1]),
        "price": p_prev,
        "long":  bool(long_sig),
        "short": bool(short_sig),
        "filter": f_prev,
        "hi": float(hi.iloc[-1]),
        "lo": float(lo.iloc[-1]),
    }

# =================== ZONES / COUNCIL ===================
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

def touch_and_reject(df: pd.DataFrame, zones):
    if len(df)<2: return {"reject_supply":False,"reject_demand":False}
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
    l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
    sup=zones.get("supply"); dem=zones.get("demand")
    rej_sup=False; rej_dem=False
    if sup:
        mid=(sup["top"]+sup["bot"])/2.0
        if (h>=sup["bot"] or _near_bps(h,sup["bot"],LEVEL_NEAR_BPS)) and c<mid and (upper/rng)>=WICK_RATIO_MIN:
            rej_sup=True
    if dem:
        mid=(dem["top"]+dem["bot"])/2.0
        if (l<=dem["top"] or _near_bps(l,dem["top"],LEVEL_NEAR_BPS)) and c>mid and (lower/rng)>=WICK_RATIO_MIN:
            rej_dem=True
    return {"reject_supply":rej_sup, "reject_demand":rej_dem}

def breakout_votes(df: pd.DataFrame, ind: dict, zones: dict):
    """يعطي أصوات عند اختراق قوي للصناديق."""
    b_votes = s_votes = 0
    b_reas  = [];  s_reas = []
    if len(df) < 2: 
        return b_votes, b_reas, s_votes, s_reas
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
    l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
    atr=float(ind.get("atr") or 0.0)
    adx=float(ind.get("adx") or 0.0)
    pdi=float(ind.get("plus_di") or 0.0)
    mdi=float(ind.get("minus_di") or 0.0)
    body = abs(c-o)
    body_ok = (atr>0) and (body >= BREAK_BODY_ATR_MIN * atr)
    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    sup = zones.get("supply"); dem = zones.get("demand")
    if sup and sup.get("top"):
        above = (c > sup["top"]) and (_bps(c, sup["top"]) >= BREAK_HYST_BPS)
        trend_ok = (adx >= BREAK_ADX_MIN) and (pdi >= mdi + BREAK_DI_MARGIN)
        if above and body_ok and trend_ok:
            b_votes += 2; b_reas.append("breakout@supply +2")
    if dem and dem.get("bot"):
        below = (c < dem["bot"]) and (_bps(c, dem["bot"]) >= BREAK_HYST_BPS)
        trend_ok = (adx >= BREAK_ADX_MIN) and (mdi >= pdi + BREAK_DI_MARGIN)
        if below and body_ok and trend_ok:
            s_votes += 2; s_reas.append("breakout@demand +2")
    return b_votes, b_reas, s_votes, s_reas

def countertrend_signal(df: pd.DataFrame, ind: dict, zones: dict, info: dict):
    """إشارة مضاد ترند عند مناطق قوية (اختياري)."""
    if not COUNTER_ENABLE or len(df)<1: return None
    adx = float(ind.get("adx") or 0.0)
    if adx >= CT_ADX_MAX: return None
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
    l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
    sup=zones.get("supply"); dem=zones.get("demand")
    try:
        hyst_bps = abs((info["price"]-info["filter"])/info["filter"])*10000.0
    except Exception:
        hyst_bps = 0.0
    # SELL near supply
    if sup and (_near_bps(h, sup["bot"], SUPPLY_TOUCH_BPS)) and (upper/rng)>=WICK_RATIO_MIN and hyst_bps>=CT_HYST_MIN_BPS:
        return {"side":"sell","votes":2,"reason":"counter@supply (wick+near+stretched)"}
    # BUY near demand
    if dem and (_near_bps(l, dem["top"], DEMAND_TOUCH_BPS)) and (lower/rng)>=WICK_RATIO_MIN and hyst_bps>=CT_HYST_MIN_BPS:
        return {"side":"buy","votes":2,"reason":"counter@demand (wick+near+stretched)"}
    return None

def council_votes(df, ind, info, zones):
    """يعيد (buy_votes, reasons_b, sell_votes, reasons_s, score_b, score_s)."""
    reasons_b=[]; reasons_s=[]; b=s=0
    score_b=0.0; score_s=0.0

    # 0) Breakout قوي (تصويت مضاعف + رفع Score)
    bk_b, bk_br, bk_s, bk_sr = breakout_votes(df, ind, zones)
    if bk_b>0: b+=bk_b; reasons_b += bk_br; score_b += 1.5
    if bk_s>0: s+=bk_s; reasons_s += bk_sr; score_s += 1.5

    # 1) Counter-trend قوي (اختياري)
    ct = countertrend_signal(df, ind, zones, info)
    if ct:
        if ct["side"]=="buy": b+=ct["votes"]; reasons_b.append(ct["reason"]); score_b += 1.0
        else: s+=ct["votes"]; reasons_s.append(ct["reason"]); score_s += 1.0

    # 2) رفض منطقة
    rej = touch_and_reject(df, zones)
    if rej["reject_demand"]: b+=1; reasons_b.append("reject@demand"); score_b += 0.5
    if rej["reject_supply"]: s+=1; reasons_s.append("reject@supply"); score_s += 0.5

    # 3) RF closed
    if info.get("long"):  b+=1; reasons_b.append("rf_long");  score_b += 0.5
    if info.get("short"): s+=1; reasons_s.append("rf_short"); score_s += 0.5

    # 4) DI/ADX
    pdi, mdi, adx = ind.get("plus_di",0), ind.get("minus_di",0), ind.get("adx",0)
    if pdi>mdi and adx>=18: b+=1; reasons_b.append("DI+>DI- & ADX"); score_b += 0.5
    if mdi>pdi and adx>=18: s+=1; reasons_s.append("DI->DI+ & ADX"); score_s += 0.5

    # 5) RSI محايد يساند لون الشمعة
    rsi = ind.get("rsi",50)
    if RSI_NEUTRAL_MIN <= rsi <= RSI_NEUTRAL_MAX:
        if float(df["close"].iloc[-1])>float(df["open"].iloc[-1]):
            b+=1; reasons_b.append("RSI_neutral_up"); score_b += 0.25
        else:
            s+=1; reasons_s.append("RSI_neutral_down"); score_s += 0.25

    # 6) شموع
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
    l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
    if (lower/rng)>=0.6 and c>o: b+=1; reasons_b.append("hammer_like"); score_b += 0.25
    if (upper/rng)>=0.6 and c<o: s+=1; reasons_s.append("shooting_like"); score_s += 0.25

    # score baseline = votes/4 + weighted extras
    score_b += b/4.0
    score_s += s/4.0
    return b, reasons_b, s, reasons_s, float(score_b), float(score_s)

# =================== ORDERING ===================
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
            if not _sym_match(sym, SYMBOL): 
                continue
            qty = None
            for key in ("contracts","size","positionAmt"):
                v = p.get(key) or p.get("info",{}).get(key)
                if v is not None:
                    try: qty = abs(float(v)); break
                    except Exception: pass
            if not qty or qty <= 0: 
                continue
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0) or 0.0
            side_raw = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            cost = float(p.get("cost") or p.get("info",{}).get("positionValue") or 0)
            side = "long" if ("long" in side_raw or cost>=0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position error: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    effective = balance or 0.0
    capital = effective * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def open_market(side, qty, price, strength, reason):
    if qty<=0:
        print(colored("❌ skip open (qty<=0)", "red")); return False
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        except Exception as e:
            print(colored(f"❌ open: {e}", "red")); logging.error(f"open_market error: {e}"); return False
    STATE.update({
        "open": True, "side": "long" if side=="buy" else "short", "entry": price,
        "qty": qty, "pnl": 0.0, "bars": 0, "trail": None,
        "hp_pct": 0.0, "strength": float(strength)
    })
    TRADE_TIMES.append(time.time())
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)} | strength={fmt(strength,2)} | {reason}", "green" if side=="buy" else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price} strength={strength} reason={reason}")
    return True

def _reset_after_close(reason, prev_side=None):
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None,
        "hp_pct": 0.0, "strength": 0.0
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
            time.sleep(1.6)
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
                _reset_after_close(reason, prev_side=side)
                LAST_CLOSE_TS = time.time()
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            print(colored(f"⚠️ strict close retry {attempts}/6 — residual={fmt(left_qty,4)}","yellow"))
            time.sleep(1.6)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(1.6)
    print(colored(f"❌ STRICT CLOSE FAILED — last error: {last_error}", "red"))

# =================== ONE-SHOT MANAGEMENT ===================
def _wick_long_enough(df: pd.DataFrame, side: str):
    if len(df)<1: return False
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
    l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
    return (upper/rng)>=WICK_RATIO_MIN if side=="long" else (lower/rng)>=WICK_RATIO_MIN

def _opp_rf_confirmed(ind: dict, info: dict, side: str) -> bool:
    try:
        px = info.get("price"); rf = info.get("filter")
        hyst = abs((px-rf)/rf)*10000.0 if (px and rf) else 0.0
        adx = float(ind.get("adx") or 0.0)
        if side=="long" and info.get("short"): return (adx>=OPP_RF_MIN_ADX) and (hyst>=OPP_RF_MIN_HYST_BPS)
        if side=="short" and info.get("long"): return (adx>=OPP_RF_MIN_ADX) and (hyst>=OPP_RF_MIN_HYST_BPS)
    except Exception: pass
    return False

def apex_confirmed(side: str, df: pd.DataFrame, ind: dict, zones: dict):
    try:
        o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
        l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
        rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
        sup=zones.get("supply"); dem=zones.get("demand")
        adx=float(ind.get("adx") or 0.0); rsi=float(ind.get("rsi") or 50.0)
        near_top = sup and (_near_bps(h, sup["bot"], 10.0))
        near_bot = dem and (_near_bps(l, dem["top"], 10.0))
        reject_top = (upper/rng>=0.55 and (adx<20 or 45<=rsi<=55))
        reject_bot = (lower/rng>=0.55 and (adx<20 or 45<=rsi<=55))
        if side=="long" and near_top and reject_top:  return True
        if side=="short" and near_bot and reject_bot: return True
    except Exception: pass
    return False

def manage_position(df, ind, info, zones):
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr = (px - entry)/entry*100*(1 if side=="long" else -1)

    # Apex rejection
    if apex_confirmed(side, df, ind, zones):
        close_market_strict("APEX_CONFIRMED"); return

    # Long wick + decent PnL
    if rr >= WICK_TAKE_MIN_PCT and _wick_long_enough(df, side):
        close_market_strict("LONG_WICK_TAKE"); return

    # Opposite RF confirmed
    if _opp_rf_confirmed(ind, info, side):
        close_market_strict("OPPOSITE_RF_CONFIRMED"); return

    # ATR trail (trend ride)
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

# =================== SIGNAL EVALUATION & ARBITRATION ===================
def evaluate_signals(df):
    info = rf_signal_closed(df)
    ind  = compute_indicators(df)
    zones = detect_zones(df)
    b,b_r,s,s_r,score_b,score_s = council_votes(df, ind, info, zones)

    candidates = []
    # Council BUY
    if b >= COUNCIL_ENTRY_VOTES_MIN:
        candidates.append({"side":"buy","score":score_b,"reason":f"Council BUY {b} votes :: {b_r}"})
    # Council SELL
    if s >= COUNCIL_ENTRY_VOTES_MIN:
        candidates.append({"side":"sell","score":score_s,"reason":f"Council SELL {s} votes :: {s_r}"})
    # RF-only as fallback candidate (وزن أقل)
    if info.get("long"):
        candidates.append({"side":"buy","score":1.0,"reason":"RF_LONG"})
    if info.get("short"):
        candidates.append({"side":"sell","score":1.0,"reason":"RF_SHORT"})

    # pick best by score (tie-breaker: use DI/ADX direction)
    if candidates:
        candidates.sort(key=lambda x: x["score"], reverse=True)

    # For metrics visibility
    return info, ind, zones, candidates

def choose_best_entry(candidates, ind):
    if not candidates: return None
    best = candidates[0]
    # تعزيز للكفة الموافقة للـDI
    pdi=float(ind.get("plus_di") or 0.0); mdi=float(ind.get("minus_di") or 0.0)
    if len(candidates)>=2 and candidates[0]["score"]==candidates[1]["score"]:
        if best["side"]=="buy" and pdi<mdi: best = candidates[1]
        if best["side"]=="sell" and mdi<pdi: best = candidates[1]
    return best

# =================== SNAPSHOT ===================
def pretty_snapshot(bal, info, ind, spread_bps, zones, reason=None, df=None, cand=None):
    left_s = time_to_candle_close(df) if df is not None else 0
    print(colored("─"*110,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
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
        print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  HP={fmt(STATE['hp_pct'],2)}%  Strength={fmt(STATE['strength'],2)}")
    else:
        print("   ⚪ FLAT")
    if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
    print(colored("─"*110,"cyan"))

# =================== MAIN LOOP ===================
def trade_loop():
    global LAST_CLOSE_TS
    loop_i=0
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()

            info, ind, zones, candidates = evaluate_signals(df)
            best = choose_best_entry(candidates, ind)

            # Guards
            spread_bps = orderbook_spread_bps()
            reason=None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            if not STATE["open"] and reason is None and float(ind.get("adx") or 0.0) < ADX_ENTRY_MIN:
                reason = f"ADX<{int(ADX_ENTRY_MIN)}: pause entries"
            if reason is None and (time.time() - LAST_CLOSE_TS) < COOLDOWN_SEC:
                reason = f"cooldown {(COOLDOWN_SEC - int(time.time()-LAST_CLOSE_TS))}s"
            while TRADE_TIMES and time.time()-TRADE_TIMES[0] > 3600:
                TRADE_TIMES.popleft()
            if reason is None and len(TRADE_TIMES) >= MAX_TRADES_PER_HOUR:
                reason = "rate-limit: too many trades this hour"

            # Update PnL if open
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]

            # Manage current position
            manage_position(df, ind, {"price": px or info["price"], **info}, zones)

            # If open and opposite stronger decision arises -> close & flip to stronger
            if STATE["open"] and best and reason is None:
                # if best side opposite AND best score > current strength + margin
                margin = 0.5
                opposite = (best["side"]=="buy" and STATE["side"]=="short") or (best["side"]=="sell" and STATE["side"]=="long")
                if opposite and best["score"] >= max(STATE.get("strength",0.0)+margin, COUNCIL_STRONG_SCORE_MIN):
                    close_market_strict("REVERSE_TO_STRONGER_DECISION")
                    # after strict close, re-evaluate price and open immediately
                    bal = balance_usdt(); px = price_now()
                    qty = compute_size(bal, px or info["price"])
                    if qty>0: open_market("buy" if best["side"]=="buy" else "sell", qty, px or info["price"], best["score"], best["reason"])

            # ENTRY: if flat and best candidate available and no guard reason
            if (not STATE["open"]) and best and reason is None:
                qty = compute_size(bal, px or info["price"])
                if qty>0 and open_market("buy" if best["side"]=="buy" else "sell", qty, px or info["price"], best["score"], best["reason"]):
                    pass
                else:
                    reason="qty<=0 or open failed"

            pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, zones, reason, df, best)

            # Bars counter
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"] += 1

            loop_i += 1
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
    return f"✅ BYBIT SOL BOT — {SYMBOL} {INTERVAL} — {mode} — RF CLOSED + Smart Council • Single-Entry Arbitration"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_gate": {"adx_min": ADX_ENTRY_MIN, "max_spread_bps": MAX_SPREAD_BPS},
        "council": {"votes_min": COUNCIL_ENTRY_VOTES_MIN, "strong_score_min": COUNCIL_STRONG_SCORE_MIN}
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "hp_pct": STATE.get("hp_pct", 0.0), "strength": STATE.get("strength",0.0)
    }), 200

def keepalive_loop():
    url=SELF_URL.strip().rstrip("/")
    if not url:
        print(colored("⏸️ keepalive disabled (SELF_URL not set)", "yellow"))
        return
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
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  ENTRY via Arbitration (Council>RF)", "yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
