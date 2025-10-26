# -*- coding: utf-8 -*-
"""
BYBIT — SOL Perp Bot (RF-LIVE + Council)
• Exchange: Bybit USDT Perps via CCXT
• Symbol: SOL/USDT:USDT (قابل للتغيير من ENV)
• Entry: RF (live candle) + Council (اصطياد قاع/قمة مؤكد)
• Post-entry: TP1 ديناميكي + Breakeven + ATR-Trail + Apex take
• Strict close + reduceOnly + Dust guard
• Opposite-RF defense (partial + tighten) + votes
• Cooldown + rate-limit لمنع الفليب-فلوب
• Flask /metrics /health + logging
"""

import os, time, math, random, signal, sys, traceback, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from collections import deque
import pandas as pd
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV / MODE ===================
API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")
MODE_LIVE = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", 5000))

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "SOL/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")  # oneway/hedge

# RF (TradingView-like) — live candle only
RF_SOURCE = os.getenv("RF_SOURCE", "close")
RF_PERIOD = int(os.getenv("RF_PERIOD", 20))
RF_MULT   = float(os.getenv("RF_MULT", 3.5))
RF_LIVE_ONLY = str(os.getenv("RF_LIVE_ONLY","true")).lower()=="true"
RF_HYST_BPS  = float(os.getenv("RF_HYST_BPS", 6.0))  # hysteresis لتقليل الذبذبة

# Indicators
RSI_LEN = int(os.getenv("RSI_LEN", 14))
ADX_LEN = int(os.getenv("ADX_LEN", 14))
ATR_LEN = int(os.getenv("ATR_LEN", 14))

# Spread guard
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 8.0))

# Dynamic TP / trail
TP1_PCT_BASE       = float(os.getenv("TP1_PCT_BASE", 0.40))
TP1_CLOSE_FRAC     = float(os.getenv("TP1_CLOSE_FRAC", 0.50))
BREAKEVEN_AFTER    = float(os.getenv("BREAKEVEN_AFTER", 0.30))
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT", 1.20))
ATR_TRAIL_MULT     = float(os.getenv("ATR_TRAIL_MULT", 1.6))

# Dust / final-chunk guard
FINAL_CHUNK_QTY    = float(os.getenv("FINAL_CHUNK_QTY", 0.2))
RESIDUAL_MIN_QTY   = float(os.getenv("RESIDUAL_MIN_QTY", 0.05))

# Opposite-RF confirmation (defense)
OPP_RF_VOTES_NEEDED = int(os.getenv("OPP_RF_VOTES_NEEDED", 2))
OPP_RF_MIN_ADX      = float(os.getenv("OPP_RF_MIN_ADX", 22.0))
OPP_RF_MIN_HYST_BPS = float(os.getenv("OPP_RF_MIN_HYST_BPS", 8.0))

# Council params (قابلة للضبط من ENV)
COUNCIL_MIN_VOTES_FOR_ENTRY = int(os.getenv("COUNCIL_MIN_VOTES_FOR_ENTRY", 4))  # out of 6
LEVEL_NEAR_BPS              = float(os.getenv("LEVEL_NEAR_BPS", 12.0))
RSI_NEUTRAL_MIN             = float(os.getenv("RSI_NEUTRAL_MIN", 45.0))
RSI_NEUTRAL_MAX             = float(os.getenv("RSI_NEUTRAL_MAX", 55.0))
ADX_COOL_OFF_DROP           = float(os.getenv("ADX_COOL_OFF_DROP", 2.0))
RETEST_MAX_BARS             = int(os.getenv("RETEST_MAX_BARS", 8))

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# Anti flip-flop
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", 90))
MAX_TRADES_PER_HOUR = int(os.getenv("MAX_TRADES_PER_HOUR", 6))

# =================== LOGGING ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready", "cyan"))

setup_file_logging()

# =================== EXCHANGE (Bybit) ===================
def make_ex():
    return ccxt.bybit({
        "apiKey": API_KEY,
        "secret": API_SECRET,
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

# =================== HELPERS / STATE ===================
_consec_err = 0
last_loop_ts = time.time()
LAST_CLOSE_TS = 0
TRADE_TIMES = deque(maxlen=10)

compound_pnl = 0.0
wait_for_next_signal_side = None  # after close wait opposite RF

STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0, "opp_votes": 0,
    "fusion_score": 0.0, "trap_risk": 0.0
}

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC>=0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
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

def with_retry(fn, tries=3, base_wait=0.4):
    global _consec_err
    for i in range(tries):
        try:
            r = fn()
            _consec_err = 0
            return r
        except Exception:
            _consec_err += 1
            if i == tries-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.25)

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
    left = max(0, next_close_ms - now_ms)
    return int(left/1000)

# =================== INDICATORS / RF ===================
def wilder_ema(s: pd.Series, n: int):
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
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

def rf_signal_live(df: pd.DataFrame):
    if len(df) < RF_PERIOD + 3:
        i = -1
        price = float(df["close"].iloc[i]) if len(df) else None
        return {"time": int(df["time"].iloc[i]) if len(df) else int(time.time()*1000),
                "price": price or 0.0, "long": False, "short": False,
                "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0}
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))

    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0

    p_now = float(src.iloc[-1]); p_prev = float(src.iloc[-2])
    f_now = float(filt.iloc[-1]); f_prev = float(filt.iloc[-2])

    long_flip  = (p_prev <= f_prev and p_now > f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    short_flip = (p_prev >= f_prev and p_now < f_now and _bps(p_now, f_now) >= RF_HYST_BPS)

    return {
        "time": int(df["time"].iloc[-1]), "price": p_now,
        "long": bool(long_flip), "short": bool(short_flip),
        "filter": f_now, "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])
    }

# =================== ZONES (Supply/Demand) + Council ===================
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
            sup={"side":"supply","top":top,"bot":bot}
        if lows:
            bot=min(lows); top=min(lows) + (max(lows)-min(lows))*0.25 if len(lows)>1 else bot*1.002
            dem={"side":"demand","top":top,"bot":bot}
        return {"supply":sup, "demand":dem}
    except Exception:
        return {"supply":None, "demand":None}

def _near(px, lvl, bps):
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
        if (h>=sup["bot"] or _near(h,sup["bot"],LEVEL_NEAR_BPS)) and c<mid and (upper/rng)>=0.5:
            rej_sup=True
    if dem:
        mid=(dem["top"]+dem["bot"])/2.0
        if (l<=dem["top"] or _near(l,dem["top"],LEVEL_NEAR_BPS)) and c>mid and (lower/rng)>=0.5:
            rej_dem=True
    return {"reject_supply":rej_sup, "reject_demand":rej_dem}

def council_votes(df, ind, info, zones):
    reasons_b=[]; reasons_s=[]; b=s=0
    rej = touch_and_reject(df, zones)
    if rej["reject_demand"]: b+=1; reasons_b.append("reject@demand")
    if rej["reject_supply"]: s+=1; reasons_s.append("reject@supply")
    if info.get("long"):  b+=1; reasons_b.append("rf_long")
    if info.get("short"): s+=1; reasons_s.append("rf_short")
    pdi, mdi, adx = ind.get("plus_di",0), ind.get("minus_di",0), ind.get("adx",0)
    if pdi>mdi and adx>=18: b+=1; reasons_b.append("DI+>DI- & ADX")
    if mdi>pdi and adx>=18: s+=1; reasons_s.append("DI->DI+ & ADX")
    rsi = ind.get("rsi",50)
    if RSI_NEUTRAL_MIN <= rsi <= RSI_NEUTRAL_MAX:
        if float(df["close"].iloc[-1])>float(df["open"].iloc[-1]):
            b+=1; reasons_b.append("RSI_neutral_up")
        else:
            s+=1; reasons_s.append("RSI_neutral_down")
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1]); l=float(df["low"].iloc[-1]); c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
    if (lower/rng)>=0.6 and c>o: b+=1; reasons_b.append("hammer_like")
    if (upper/rng)>=0.6 and c<o: s+=1; reasons_s.append("shooting_like")
    return b, reasons_b, s, reasons_s

# =================== ORDERS ===================
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
            if SYMBOL.split(":")[0] not in sym: continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty <= 0: return 0.0, None, None
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side = "long" if ("long" in side_raw or float(p.get("cost",0))>0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position error: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    effective = balance or 0.0
    capital = effective * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def open_market(side, qty, price):
    if qty<=0:
        print(colored("❌ skip open (qty<=0)", "red"))
        return False
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        except Exception as e:
            print(colored(f"❌ open: {e}", "red"))
            logging.error(f"open_market error: {e}")
            return False
    STATE.update({
        "open": True, "side": "long" if side=="buy" else "short", "entry": price,
        "qty": qty, "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "opp_votes": 0
    })
    TRADE_TIMES.append(time.time())
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)}", "green" if side=="buy" else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price}")
    return True

def close_market_strict(reason="STRICT"):
    global compound_pnl, wait_for_next_signal_side, LAST_CLOSE_TS
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0:
        if STATE.get("open"):
            _reset_after_close(reason)
        return
    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts=0; last_error=None
    while attempts < 6:
        try:
            if MODE_LIVE:
                params = _params_close(); params["reduceOnly"]=True
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(1.8)
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
            time.sleep(1.8)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(1.8)
    print(colored(f"❌ STRICT CLOSE FAILED — last error: {last_error}", "red"))

def _reset_after_close(reason, prev_side=None):
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "opp_votes": 0
    })
    if prev_side == "long":  wait_for_next_signal_side = "sell"
    elif prev_side == "short": wait_for_next_signal_side = "buy"
    else: wait_for_next_signal_side = None
    logging.info(f"AFTER_CLOSE reason={reason} wait_for={wait_for_next_signal_side}")

def close_partial(frac, reason):
    if not STATE["open"] or STATE["qty"]<=0: return
    qty_close = safe_qty(max(0.0, STATE["qty"] * min(max(frac,0.0),1.0)))
    px = price_now() or STATE["entry"]
    min_unit = max(RESIDUAL_MIN_QTY, LOT_MIN or RESIDUAL_MIN_QTY)
    if qty_close < min_unit:
        print(colored(f"⏸️ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(min_unit,4)})", "yellow"))
        return
    side = "sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close())
        except Exception as e: print(colored(f"❌ partial close: {e}", "red")); return
    pnl = (px - STATE["entry"]) * qty_close * (1 if STATE["side"]=="long" else -1)
    STATE["qty"] = safe_qty(STATE["qty"] - qty_close)
    logging.info(f"PARTIAL_CLOSE {reason} qty={qty_close} pnl={pnl} rem={STATE['qty']}")
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(STATE['qty'],4)}","magenta"))
    if STATE["qty"] <= FINAL_CHUNK_QTY and STATE["qty"]>0:
        print(colored(f"🧹 Final chunk ≤ {FINAL_CHUNK_QTY} → strict close", "yellow"))
        close_market_strict("FINAL_CHUNK_RULE")

# =================== DEFENSE ON OPPOSITE RF ===================
def defensive_on_opposite_rf(ind: dict, info: dict):
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info.get("price") or price_now() or STATE["entry"]
    rf = info.get("filter")
    adx=float(ind.get("adx") or 0.0)
    base_frac = 0.25 if not STATE.get("tp1_done") else 0.20
    close_partial(base_frac, "Opposite RF — defensive")
    if STATE.get("breakeven") is None: STATE["breakeven"]=STATE["entry"]
    atr=float(ind.get("atr") or 0.0)
    if atr>0 and px is not None:
        gap = atr * max(ATR_TRAIL_MULT, 1.2)
        if STATE["side"]=="long":
            STATE["trail"]=max(STATE["trail"] or (px-gap), px-gap)
        else:
            STATE["trail"]=min(STATE["trail"] or (px+gap), px+gap)
    STATE["opp_votes"]=int(STATE.get("opp_votes",0))+1
    hyst=0.0
    try:
        if px and rf: hyst = abs((px-rf)/rf)*10000.0
    except Exception: pass
    votes_ok = STATE["opp_votes"]>=OPP_RF_VOTES_NEEDED
    confirmed = (adx>=OPP_RF_MIN_ADX) and (hyst>=OPP_RF_MIN_HYST_BPS)
    if votes_ok and confirmed:
        close_market_strict("OPPOSITE_RF_CONFIRMED")

# =================== TP / APEX / TRAIL ===================
def _consensus(ind, info, side) -> float:
    score=0.0
    try:
        adx=float(ind.get("adx") or 0.0)
        rsi=float(ind.get("rsi") or 50.0)
        if (side=="long" and rsi>=55) or (side=="short" and rsi<=45): score += 1.0
        if adx>=28: score += 1.0
        elif adx>=20: score += 0.5
        if abs(info["price"]-info["filter"])/max(info["filter"],1e-9) >= (RF_HYST_BPS/10000.0): score += 0.5
    except Exception: pass
    return float(score)

def _tp_ladder(info, ind, side):
    px = info["price"]; atr = float(ind.get("atr") or 0.0)
    atr_pct = (atr / max(px,1e-9))*100.0 if px else 0.5
    score = _consensus(ind, info, side)
    if score >= 2.5: mults = [1.8, 3.2, 5.0]
    elif score >= 1.5: mults = [1.6, 2.8, 4.5]
    else: mults = [1.2, 2.4, 4.0]
    tps = [round(m*atr_pct, 2) for m in mults]
    frs = [0.25, 0.30, 0.45]
    return tps, frs

def apex_confirmed(side: str, df: pd.DataFrame, ind: dict, zones: dict):
    try:
        o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
        l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
        rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
        sup=zones.get("supply"); dem=zones.get("demand")
        adx=float(ind.get("adx") or 0.0); rsi=float(ind.get("rsi") or 50.0)
        near_top = sup and (_near(h, sup["bot"], 10.0))
        near_bot = dem and (_near(l, dem["top"], 10.0))
        reject_top = (upper/rng>=0.55 and (adx<20 or 45<=rsi<=55))
        reject_bot = (lower/rng>=0.55 and (adx<20 or 45<=rsi<=55))
        if side=="long" and near_top and reject_top:  return True
        if side=="short" and near_bot and reject_bot: return True
    except Exception: pass
    return False

def manage_after_entry(df, ind, info, zones):
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr = (px - entry)/entry*100*(1 if side=="long" else -1)

    dyn_tps, dyn_fracs = _tp_ladder(info, ind, side)
    STATE["_tp_cache"]=dyn_tps; STATE["_tp_fracs"]=dyn_fracs
    k = int(STATE.get("profit_targets_achieved", 0))

    tp1_now = TP1_PCT_BASE*(2.2 if ind.get("adx",0)>=35 else 1.8 if ind.get("adx",0)>=28 else 1.0)
    if (not STATE["tp1_done"]) and rr >= tp1_now:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{tp1_now:.2f}%")
        STATE["tp1_done"]=True
        if rr >= BREAKEVEN_AFTER: STATE["breakeven"]=entry

    if rr >= max(0.4, TP1_PCT_BASE*0.8) and apex_confirmed(side, df, ind, zones):
        close_market_strict("APEX_CONFIRMED_FULL_TAKE"); return

    hold_explosion = False  # (تبسيط: ممكن توصلها بمؤشر انفجار لاحقًا)
    if k < len(dyn_tps) and rr >= dyn_tps[k] and not hold_explosion:
        frac = dyn_fracs[k] if k < len(dyn_fracs) else 0.25
        close_partial(frac, f"TP_dyn@{dyn_tps[k]:.2f}%")
        STATE["profit_targets_achieved"] = k + 1

    if rr > STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr

    atr=float(ind.get("atr") or 0.0)
    if rr >= TRAIL_ACTIVATE_PCT and atr>0:
        gap = atr * ATR_TRAIL_MULT
        if side=="long":
            new_trail = px - gap
            STATE["trail"] = max(STATE["trail"] or new_trail, new_trail)
            if STATE["breakeven"] is not None: STATE["trail"] = max(STATE["trail"], STATE["breakeven"])
            if px < STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")
        else:
            new_trail = px + gap
            STATE["trail"] = min(STATE["trail"] or new_trail, new_trail)
            if STATE["breakeven"] is not None: STATE["trail"] = min(STATE["trail"], STATE["breakeven"])
            if px > STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")

# =================== SNAPSHOT ===================
def pretty_snapshot(bal, info, ind, spread_bps, zones, reason=None, df=None):
    left_s = time_to_candle_close(df) if df is not None else 0
    print(colored("─"*110,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
    print("📈 RF & INDICATORS")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))} hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)} bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
    print(f"   🏗️ ZONES: {zones}")
    print(f"   ⏱️ closes_in ≈ {left_s}s")

    print("\n🧭 POSITION")
    bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(colored(f"   {bal_line}", "yellow"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
        print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%  OppVotes={STATE.get('opp_votes',0)}")
    else:
        print("   ⚪ FLAT")
        if wait_for_next_signal_side:
            print(colored(f"   ⏳ Waiting RF opposite: {wait_for_next_signal_side.upper()}", "cyan"))
    if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
    print(colored("─"*110,"cyan"))

# =================== COUNCIL DECISION WRAPPER ===================
def council_decide(df):
    info = rf_signal_live(df)
    ind  = compute_indicators(df)
    zones = detect_zones(df)

    # votes
    b,b_r,s,s_r = council_votes(df, ind, info, zones)
    entry=None; exit_sig=None

    # ENTRY (مجلس الإدارة له الأولوية مع RF)
    if not STATE["open"]:
        if b >= COUNCIL_MIN_VOTES_FOR_ENTRY:
            entry = {"side": "buy", "reason": f"council {b}✓ :: {b_r}"}
        elif s >= COUNCIL_MIN_VOTES_FOR_ENTRY:
            entry = {"side": "sell", "reason": f"council {s}✓ :: {s_r}"}

    # EXIT (رفض قوي في الصندوق المقابل)
    rej = touch_and_reject(df, zones)
    if STATE["open"]:
        if STATE["side"]=="long" and rej["reject_supply"]:
            exit_sig = {"reason":"reject@supply (hard close)"}
        if STATE["side"]=="short" and rej["reject_demand"]:
            exit_sig = {"reason":"reject@demand (hard close)"}

    return info, ind, zones, entry, exit_sig

# =================== LOOP ===================
def trade_loop():
    global wait_for_next_signal_side
    loop_i=0
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()

            info, ind, zones, council_entry, council_exit = council_decide(df)

            # spread guard
            spread_bps = orderbook_spread_bps()
            reason=None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"

            # cooldown / rate-limit
            if reason is None and (time.time() - LAST_CLOSE_TS) < COOLDOWN_SEC:
                reason = f"cooldown {(COOLDOWN_SEC - int(time.time()-LAST_CLOSE_TS))}s"
            while TRADE_TIMES and time.time()-TRADE_TIMES[0] > 3600:
                TRADE_TIMES.popleft()
            if reason is None and len(TRADE_TIMES) >= MAX_TRADES_PER_HOUR:
                reason = "rate-limit: too many trades this hour"

            # update PnL
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]

            # manage after entry
            manage_after_entry(df, ind, {"price": px or info["price"], **info}, zones)

            # defense on opposite RF while in trade
            if STATE["open"]:
                if STATE["side"]=="long" and info["short"]:
                    defensive_on_opposite_rf(ind, {"price": px or info["price"], **info})
                elif STATE["side"]=="short" and info["long"]:
                    defensive_on_opposite_rf(ind, {"price": px or info["price"], **info})

            # strict exit by council
            if STATE["open"] and council_exit:
                close_market_strict(f"COUNCIL_EXIT — {council_exit['reason']}")

            # ENTRY: Council-led (includes RF vote). Respect wait_for_next_signal after a close
            if not STATE["open"] and council_entry and reason is None:
                sig = council_entry["side"]
                if wait_for_next_signal_side and sig != wait_for_next_signal_side:
                    reason=f"waiting opposite RF: need {wait_for_next_signal_side.upper()}"
                else:
                    qty = compute_size(bal, px or info["price"])
                    if qty>0 and open_market("buy" if sig=="buy" else "sell", qty, px or info["price"]):
                        wait_for_next_signal_side=None
                        print(colored(f"[COUNCIL ENTRY] {sig.upper()} :: {council_entry['reason']}", "cyan"))
                    else:
                        reason="qty<=0 or open failed"

            pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, zones, reason, df)

            # bar counter
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
    return f"✅ BYBIT SOL BOT — {SYMBOL} {INTERVAL} — {mode} — RF Live + Council • TP/BE/Trail • Strict Close"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "RF_LIVE+COUNCIL", "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY}
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "RF_LIVE+COUNCIL", "wait_for_next_signal": wait_for_next_signal_side,
        "tp_done": STATE.get("profit_targets_achieved", 0), "opp_votes": STATE.get("opp_votes",0)
    }), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (SELF_URL not set)", "yellow"))
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
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  RF_LIVE={RF_LIVE_ONLY}", "yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
