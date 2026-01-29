import os
import sys
import time
import math
from dataclasses import dataclass
from typing import Optional, Deque, Tuple, Dict
from collections import deque

import numpy as np
import pandas as pd

import ccxt
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl


# =========================
# HARD RULES
# =========================
MIN_WINRATE = 0.90         # required on TEST and Walk-forward OOS
MIN_TRADES_TEST = 80       # anti-cheat: must have enough executed trades on TEST
MIN_TRADES_WALK = 120      # anti-cheat: enough trades across walk-forward OOS


# =========================
# PROFILES (more data + tuning grids)
# =========================
PROFILES = {
    "SCALPING (1m/3m)": {
        "primary_tf": "1m",
        "secondary_tf": "3m",
        "poll_s": 2.0,
        "days_primary": 75,
        "days_secondary": 160,

        # base values (tuning will vary these)
        "horizon_bars": 6,
        "thr": 0.0012,
        "epochs": 14,

        "train_window_frac": 0.60,
        "val_window_frac": 0.20,
        "test_window_frac": 0.20,

        "walk_train_bars": 5200,
        "walk_test_bars": 650,

        "tp_base_mult": 0.85,

        # tuning grids (continue until >= 90% winrate on executed trades)
        "thr_grid":     [0.0010, 0.0012, 0.0015, 0.0018, 0.0022, 0.0028, 0.0035],
        "horizon_grid": [4, 6, 8, 10, 12],
        "epoch_grid":   [14, 18, 24],
        "conf_grid":    [0.40, 0.45, 0.50, 0.55, 0.60, 0.65],  # trade only if prob>=conf
    },
    "SWING (15m/1h)": {
        "primary_tf": "15m",
        "secondary_tf": "1h",
        "poll_s": 5.0,
        "days_primary": 1100,
        "days_secondary": 1400,

        "horizon_bars": 8,
        "thr": 0.0025,
        "epochs": 16,

        "train_window_frac": 0.60,
        "val_window_frac": 0.20,
        "test_window_frac": 0.20,

        "walk_train_bars": 5200,
        "walk_test_bars": 520,

        "tp_base_mult": 1.25,

        "thr_grid":     [0.0018, 0.0022, 0.0028, 0.0035, 0.0045, 0.0055],
        "horizon_grid": [6, 8, 10, 12, 16],
        "epoch_grid":   [16, 22, 28],
        "conf_grid":    [0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
    }
}

SYMBOL_SPOT = "ETH/USDT"
FUT_CANDIDATES = ["ETH/USDT:USDT", "ETHUSDTM", "ETH/USDTM"]


# =========================
# UTIL
# =========================
def timeframe_to_ms(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1]) * 60_000
    if tf.endswith("h"):
        return int(tf[:-1]) * 3_600_000
    if tf.endswith("d"):
        return int(tf[:-1]) * 86_400_000
    raise ValueError(f"Unsupported timeframe: {tf}")

def safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")

def clamp_df(df: pd.DataFrame, lo=-8.0, hi=8.0) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], 0).fillna(0).clip(lo, hi)

def has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return f"GPU CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "GPU CUDA"
    return "CPU"


# =========================
# TA FEATURES (OHLCV)
# =========================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    last_close = close.iloc[-1]
    if pd.isna(atr) or last_close <= 0:
        return 0.0
    return float(atr / (last_close + 1e-12))

def features_ohlcv(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volu = df["volume"].astype(float)

    ret1 = close.pct_change().fillna(0)
    vol20 = ret1.rolling(20).std().fillna(0)
    r = rsi(close, 14)
    m_line, s_line, hist = macd(close)

    out = pd.DataFrame(index=df.index)
    out[f"{prefix}body_pct"] = (close - open_) / (open_ + 1e-12)
    out[f"{prefix}range_pct"] = (high - low) / (close + 1e-12)
    out[f"{prefix}ret1"] = ret1
    out[f"{prefix}ret5_mean"] = ret1.rolling(5).mean().fillna(0)
    out[f"{prefix}vol20"] = vol20
    out[f"{prefix}rsi14"] = (r / 100.0)
    out[f"{prefix}macd"] = m_line.fillna(0)
    out[f"{prefix}macd_sig"] = s_line.fillna(0)
    out[f"{prefix}macd_hist"] = hist.fillna(0)
    out[f"{prefix}log_price"] = np.log(close + 1e-12)
    out[f"{prefix}log_vol"] = np.log(volu + 1e-12)

    return clamp_df(out)


# =========================
# LIVE "RICH" FEATURES (Orderbook + Funding/OI)
# =========================
def orderbook_features(ob: dict, depth: int = 20) -> Dict[str, float]:
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    if len(bids) == 0 or len(asks) == 0:
        return {"ob_spread": 0.0, "ob_top_imb": 0.0, "ob_depth_imb": 0.0, "ob_bid_depth": 0.0, "ob_ask_depth": 0.0}

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    spread = (best_ask - best_bid) / (mid + 1e-12)

    top_bid_sz = float(bids[0][1])
    top_ask_sz = float(asks[0][1])
    top_imb = (top_bid_sz - top_ask_sz) / (top_bid_sz + top_ask_sz + 1e-12)

    b = bids[:depth]
    a = asks[:depth]
    bid_depth = sum(float(x[1]) for x in b)
    ask_depth = sum(float(x[1]) for x in a)
    depth_imb = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-12)

    return {
        "ob_spread": float(spread),
        "ob_top_imb": float(top_imb),
        "ob_depth_imb": float(depth_imb),
        "ob_bid_depth": float(math.log(bid_depth + 1e-12)),
        "ob_ask_depth": float(math.log(ask_depth + 1e-12)),
    }

def try_fetch_futures_metrics(ex_fut) -> Dict[str, float]:
    out = {"funding_rate": 0.0, "open_interest": 0.0}
    if ex_fut is None:
        return out

    symbol = None
    for cand in FUT_CANDIDATES:
        try:
            ex_fut.market(cand)
            symbol = cand
            break
        except Exception:
            continue

    try:
        if symbol and hasattr(ex_fut, "fetch_funding_rate"):
            fr = ex_fut.fetch_funding_rate(symbol)
            out["funding_rate"] = float(fr.get("fundingRate", fr.get("funding_rate", 0.0)) or 0.0)
    except Exception:
        pass

    try:
        if symbol and hasattr(ex_fut, "fetch_open_interest"):
            oi = ex_fut.fetch_open_interest(symbol)
            v = oi.get("openInterestAmount", oi.get("openInterest", 0.0))
            out["open_interest"] = float(v or 0.0)
    except Exception:
        pass

    out["open_interest"] = float(np.clip(out["open_interest"], 0.0, 1e12))
    out["open_interest"] = float(math.log(out["open_interest"] + 1e-12))
    out["funding_rate"] = float(np.clip(out["funding_rate"], -0.01, 0.01))
    return out


# =========================
# DATA LOADER (KuCoin OHLCV full)
# =========================
def fetch_ohlcv_full(exchange, symbol: str, timeframe: str, days: int, limit: int = 1500) -> pd.DataFrame:
    now_ms = exchange.milliseconds()
    since = now_ms - days * 86_400_000
    tf_ms = timeframe_to_ms(timeframe)

    all_rows = []
    last_ts = None

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        if last_ts is not None and batch[-1][0] <= last_ts:
            break

        last_ts = batch[-1][0]
        all_rows.extend(batch)
        since = batch[-1][0] + tf_ms

        time.sleep(exchange.rateLimit / 1000.0)

        if since >= now_ms - tf_ms:
            break

    df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# =========================
# MULTI-TF FEATURES
# =========================
def merge_primary_secondary(df_p: pd.DataFrame, df_s: pd.DataFrame, tf_s_ms: int) -> pd.DataFrame:
    fp = features_ohlcv(df_p, prefix="p_")
    fs = features_ohlcv(df_s, prefix="s_")

    p = df_p[["timestamp"]].copy()
    s = df_s[["timestamp"]].copy()
    p["ts"] = pd.to_datetime(p["timestamp"], unit="ms")
    s["ts"] = pd.to_datetime(s["timestamp"], unit="ms")

    fp2 = fp.copy(); fs2 = fs.copy()
    fp2["ts"] = p["ts"].values
    fs2["ts"] = s["ts"].values
    fp2 = fp2.sort_values("ts")
    fs2 = fs2.sort_values("ts")

    merged = pd.merge_asof(fp2, fs2, on="ts", direction="backward",
                           tolerance=pd.Timedelta(milliseconds=tf_s_ms * 2))
    merged = merged.drop(columns=["ts"]).fillna(0)
    return clamp_df(merged)


# =========================
# LABELS: 3 classes (LONG/SHORT/NO TRADE)
# 0=LONG, 1=SHORT, 2=NO_TRADE
# =========================
def build_labels_3(close: np.ndarray, horizon: int, thr: float) -> np.ndarray:
    y = np.full(len(close), 2, dtype=np.int64)  # default NO_TRADE
    for i in range(len(close) - horizon):
        now = close[i]
        future = close[i + horizon]
        if future > now * (1 + thr):
            y[i] = 0
        elif future < now * (1 - thr):
            y[i] = 1
        else:
            y[i] = 2
    # last horizon bars: keep NO_TRADE (no future)
    return y

def make_dataset_multitf_3(df_p: pd.DataFrame, df_s: pd.DataFrame,
                           horizon: int, thr: float, tf_s_ms: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feat = merge_primary_secondary(df_p, df_s, tf_s_ms=tf_s_ms)

    # placeholders for live-only features (deterministic zeros during training)
    for col in ["ob_spread", "ob_top_imb", "ob_depth_imb", "ob_bid_depth", "ob_ask_depth", "funding_rate", "open_interest"]:
        feat[col] = 0.0

    close = df_p["close"].astype(float).values
    y = build_labels_3(close, horizon=horizon, thr=thr)

    n = min(len(feat), len(y))
    feat = feat.iloc[:n].reset_index(drop=True)
    df_used = df_p.iloc[:n].reset_index(drop=True)
    y = y[:n]

    X = feat.values.astype(np.float32)
    Y = y.astype(np.int64)
    return X, Y, df_used


# =========================
# BIG MODEL (3 outputs)
# =========================
class BiggerMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, out_dim)
        self.drop = nn.Dropout(p=0.15)

        self.last_a1 = None
        self.last_a2 = None
        self.last_a3 = None
        self.last_out = None

    def forward(self, x):
        a1 = torch.tanh(self.fc1(x))
        a1d = self.drop(a1)
        a2 = torch.tanh(self.fc2(a1d))
        a2d = self.drop(a2)
        a3 = torch.tanh(self.fc3(a2d))
        out = self.fc4(a3)

        self.last_a1 = a1.detach().float().cpu().numpy()
        self.last_a2 = a2.detach().float().cpu().numpy()
        self.last_a3 = a3.detach().float().cpu().numpy()
        self.last_out = out.detach().float().cpu().numpy()
        return out


# =========================
# SPLIT / TRAIN
# =========================
def chrono_split(X: np.ndarray, y: np.ndarray, cfg: dict):
    n = len(X)
    n_train = int(n * cfg["train_window_frac"])
    n_val = int(n * cfg["val_window_frac"])
    n_test = n - n_train - n_val
    if n_train < 600 or n_val < 350 or n_test < 350:
        raise RuntimeError(f"Dataset too small for split: n={n} (train={n_train}, val={n_val}, test={n_test})")

    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    Xte, yte = X[n_train+n_val:], y[n_train+n_val:]
    return (Xtr, ytr), (Xva, yva), (Xte, yte), (n_train, n_val)

def train_one(Xtr, ytr, Xva, yva, device: torch.device, epochs: int, lr: float = 1e-3, batch_size: int = 512):
    in_dim = Xtr.shape[1]
    model = BiggerMLP(in_dim=in_dim, out_dim=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)
    Xv = torch.from_numpy(Xva).to(device)
    yv = torch.from_numpy(yva).to(device)

    best_loss = float("inf")
    best_state = None

    for _ep in range(int(epochs)):
        model.train()
        n = len(Xt)
        idx = torch.randperm(n, device=device)
        Xt_sh = Xt[idx]
        yt_sh = yt[idx]

        for start in range(0, n, batch_size):
            xb = Xt_sh[start:start+batch_size]
            yb = yt_sh[start:start+batch_size]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = float(F.cross_entropy(val_logits, yv).detach().cpu().item())

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_loss

def probs(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        Xt = torch.from_numpy(X).to(device)
        logits = model(Xt)
        return F.softmax(logits, dim=1).detach().cpu().numpy()


# =========================
# BACKTEST with NO TRADE + confidence threshold
# class mapping: 0 LONG, 1 SHORT, 2 NO_TRADE
# Decision:
#  - if argmax == 2 -> NO TRADE
#  - else if max(long,short) < conf_thr -> NO TRADE
# =========================
def backtest_3class(close: np.ndarray, probs3: np.ndarray, horizon: int, conf_thr: float, fee_bps: float = 2.0) -> dict:
    n = min(len(probs3), len(close) - horizon)
    if n <= 0:
        return {"trades": 0, "winrate": 0.0, "avg_ret": 0.0, "total": 0.0}

    fee = (fee_bps / 10_000.0)
    rets = []
    for i in range(n):
        entry = close[i]
        exitp = close[i + horizon]
        p_long, p_short, p_nt = probs3[i]

        cls = int(np.argmax([p_long, p_short, p_nt]))
        # apply confidence gating for LONG/SHORT
        if cls == 2:
            continue
        if max(p_long, p_short) < conf_thr:
            continue

        if cls == 0:  # LONG
            r = (exitp / entry) - 1.0
        else:         # SHORT
            r = (entry / exitp) - 1.0

        r = r - 2 * fee
        rets.append(r)

    if len(rets) == 0:
        return {"trades": 0, "winrate": 0.0, "avg_ret": 0.0, "total": 0.0}

    rets = np.array(rets, dtype=np.float64)
    trades = int(len(rets))
    winrate = float((rets > 0).mean())
    avg_ret = float(rets.mean())
    total = float(np.prod(1.0 + rets) - 1.0)
    return {"trades": trades, "winrate": winrate, "avg_ret": avg_ret, "total": total}


def walk_forward_oos_3class(Xs: np.ndarray, y: np.ndarray, close: np.ndarray, cfg: dict, device: torch.device, conf_thr: float, horizon: int, epochs: int) -> dict:
    train_w = int(cfg["walk_train_bars"])
    test_w = int(cfg["walk_test_bars"])

    if len(Xs) < train_w + test_w + 800:
        return {"segments": 0, "trades": 0, "winrate": 0.0, "avg_ret": 0.0, "total": 0.0}

    seg_stats = []
    start = 0
    while True:
        tr_start = start
        tr_end = tr_start + train_w
        te_start = tr_end
        te_end = te_start + test_w
        if te_end > len(Xs):
            break

        Xtr = Xs[tr_start:tr_end]
        ytr = y[tr_start:tr_end]
        cut = int(len(Xtr) * 0.85)
        Xtr2, ytr2 = Xtr[:cut], ytr[:cut]
        Xva2, yva2 = Xtr[cut:], ytr[cut:]

        model, _ = train_one(Xtr2, ytr2, Xva2, yva2, device=device, epochs=epochs, lr=1e-3, batch_size=512)
        p = probs(model, Xs[te_start:te_end], device=device)

        cseg = close[te_start:te_end]
        bt = backtest_3class(cseg, p, horizon=horizon, conf_thr=conf_thr, fee_bps=2.0)
        seg_stats.append(bt)

        start += test_w

    if not seg_stats:
        return {"segments": 0, "trades": 0, "winrate": 0.0, "avg_ret": 0.0, "total": 0.0}

    trades = sum(s["trades"] for s in seg_stats)
    if trades == 0:
        return {"segments": len(seg_stats), "trades": 0, "winrate": 0.0, "avg_ret": 0.0, "total": 0.0}

    winrate = sum(s["winrate"] * s["trades"] for s in seg_stats) / trades
    avg_ret = sum(s["avg_ret"] * s["trades"] for s in seg_stats) / trades
    total = float(np.prod([1.0 + s["total"] for s in seg_stats]) - 1.0)

    return {"segments": len(seg_stats), "trades": int(trades), "winrate": float(winrate), "avg_ret": float(avg_ret), "total": float(total)}


# =========================
# TAKE PROFIT
# =========================
def suggest_takeprofit(last_price: float, atr_percent: float, p_long: float, p_short: float, tp_base_mult: float) -> Tuple[float, float]:
    conf = abs(float(p_long) - float(p_short))
    boost = min(1.25, 0.35 + 1.2 * conf)
    tp_pct = float(max(0.0005, atr_percent * tp_base_mult * boost))
    tp_price = float(last_price * tp_pct)
    return tp_pct, tp_price


# =========================
# 3D NET VIEW (LONG green, SHORT red, NO TRADE blue)
# =========================
class Net3D(gl.GLViewWidget):
    def __init__(self, model: BiggerMLP, max_nodes_per_layer: int = 48):
        super().__init__()
        self.model = model
        self.max_nodes = max_nodes_per_layer
        self.setCameraPosition(distance=12, elevation=18, azimuth=35)

        g = gl.GLGridItem()
        g.scale(1, 1, 1)
        self.addItem(g)

        self._rebuild()

    def _layer_positions(self, n: int, x: float) -> np.ndarray:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 2.2 if n > 2 else 0.8
        y = radius * np.cos(angles)
        z = radius * np.sin(angles)
        return np.column_stack([np.full(n, x), y, z]).astype(np.float32)

    def _make_edges(self, a: np.ndarray, b: np.ndarray) -> gl.GLLinePlotItem:
        pts = []
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                pts.append(a[i]); pts.append(b[j])
        pts = np.array(pts, dtype=np.float32)
        return gl.GLLinePlotItem(pos=pts, mode="lines", width=1, antialias=True)

    def _set_gray_intensity(self, scatter: gl.GLScatterPlotItem, values: np.ndarray):
        v = np.nan_to_num(values.astype(np.float32), nan=0.0)
        v = (v - v.min()) / (v.max() - v.min() + 1e-9)
        colors = np.column_stack([v, v, v, np.full_like(v, 1.0)]).astype(np.float32)
        scatter.setData(color=colors)

    def _set_output_colors(self, p_long: float, p_short: float, p_nt: float):
        pl = float(np.clip(p_long, 0.0, 1.0))
        ps = float(np.clip(p_short, 0.0, 1.0))
        pn = float(np.clip(p_nt,   0.0, 1.0))
        c_long  = np.array([0.0, pl, 0.0, 1.0], dtype=np.float32)  # green
        c_short = np.array([ps, 0.0, 0.0, 1.0], dtype=np.float32)  # red
        c_nt    = np.array([0.0, 0.0, pn, 1.0], dtype=np.float32)  # blue
        self.s_out.setData(color=np.vstack([c_long, c_short, c_nt]).astype(np.float32))

    def _rebuild(self):
        for it in list(self.items):
            if not isinstance(it, gl.GLGridItem):
                self.removeItem(it)

        in_dim = min(self.model.fc1.in_features, self.max_nodes)
        h1 = min(self.model.fc1.out_features, self.max_nodes)
        h2 = min(self.model.fc2.out_features, self.max_nodes)
        h3 = min(self.model.fc3.out_features, self.max_nodes)

        self.pos_in = self._layer_positions(in_dim, x=-4.0)
        self.pos_h1 = self._layer_positions(h1, x=-1.6)
        self.pos_h2 = self._layer_positions(h2, x=1.0)
        self.pos_h3 = self._layer_positions(h3, x=3.3)

        # 3 outputs
        self.pos_out = np.array([[5.5,  1.2, 0.0],   # LONG
                                 [5.5,  0.0, 0.0],   # SHORT
                                 [5.5, -1.2, 0.0]],  # NO TRADE
                                dtype=np.float32)

        self.s_in = gl.GLScatterPlotItem(pos=self.pos_in, size=6)
        self.s_h1 = gl.GLScatterPlotItem(pos=self.pos_h1, size=8)
        self.s_h2 = gl.GLScatterPlotItem(pos=self.pos_h2, size=8)
        self.s_h3 = gl.GLScatterPlotItem(pos=self.pos_h3, size=8)
        self.s_out = gl.GLScatterPlotItem(pos=self.pos_out, size=12)

        self.addItem(self.s_in); self.addItem(self.s_h1); self.addItem(self.s_h2); self.addItem(self.s_h3); self.addItem(self.s_out)

        self.addItem(self._make_edges(self.pos_in, self.pos_h1))
        self.addItem(self._make_edges(self.pos_h1, self.pos_h2))
        self.addItem(self._make_edges(self.pos_h2, self.pos_h3))
        self.addItem(self._make_edges(self.pos_h3, self.pos_out))

        self._set_gray_intensity(self.s_in, np.zeros(in_dim))
        self._set_gray_intensity(self.s_h1, np.zeros(h1))
        self._set_gray_intensity(self.s_h2, np.zeros(h2))
        self._set_gray_intensity(self.s_h3, np.zeros(h3))
        self._set_output_colors(0.33, 0.33, 0.34)

    def set_model(self, model: BiggerMLP):
        self.model = model
        self._rebuild()

    def update_from_cache(self, feat_vec: np.ndarray, p_long: float, p_short: float, p_nt: float):
        inp = np.tanh(np.abs(feat_vec[:self.pos_in.shape[0]]))
        self._set_gray_intensity(self.s_in, inp)

        if self.model.last_a1 is not None:
            self._set_gray_intensity(self.s_h1, self.model.last_a1[0][:self.pos_h1.shape[0]])
        if self.model.last_a2 is not None:
            self._set_gray_intensity(self.s_h2, self.model.last_a2[0][:self.pos_h2.shape[0]])
        if self.model.last_a3 is not None:
            self._set_gray_intensity(self.s_h3, self.model.last_a3[0][:self.pos_h3.shape[0]])

        self._set_output_colors(p_long, p_short, p_nt)


# =========================
# WORKER
# =========================
@dataclass
class Tick:
    ts: float
    price: float
    long_p: float
    short_p: float
    nt_p: float
    signal: str
    conf_thr: float
    tp_pct: float
    tp_price: float
    feat: np.ndarray


class Worker(QtCore.QObject):
    status = QtCore.Signal(str)
    report = QtCore.Signal(str)
    tick = QtCore.Signal(object)
    ready = QtCore.Signal(object)   # model
    stopped = QtCore.Signal()

    def __init__(self, profile_name: str, device: torch.device):
        super().__init__()
        self.cfg = PROFILES[profile_name]
        self.profile_name = profile_name
        self.device = device
        self._stop = False

        self.ex_spot = ccxt.kucoin({"enableRateLimit": True})
        try:
            self.ex_fut = ccxt.kucoinfutures({"enableRateLimit": True})
        except Exception:
            self.ex_fut = None

        self.symbol = SYMBOL_SPOT
        self.primary_tf = self.cfg["primary_tf"]
        self.secondary_tf = self.cfg["secondary_tf"]
        self.poll_s = float(self.cfg["poll_s"])

        tag = safe_name(profile_name) + "_" + ("cuda" if device.type == "cuda" else "cpu") + "_3class"
        self.model_file = f"model_{tag}.pt"
        self.scaler_file = f"scaler_{tag}.joblib"
        self.meta_file = f"meta_{tag}.joblib"

        self.model: Optional[BiggerMLP] = None
        self.scaler: Optional[StandardScaler] = None
        self.best_conf_thr: float = 0.55
        self.best_horizon: int = int(self.cfg["horizon_bars"])
        self.best_thr: float = float(self.cfg["thr"])
        self.best_epochs: int = int(self.cfg["epochs"])

        self.window_live = 300

    def stop(self):
        self._stop = True

    def _load_if_exists(self) -> bool:
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file) and os.path.exists(self.meta_file):
            meta = joblib.load(self.meta_file)
            in_dim = int(meta["in_dim"])
            self.best_conf_thr = float(meta.get("conf_thr", self.best_conf_thr))
            self.best_horizon = int(meta.get("horizon", self.best_horizon))
            self.best_thr = float(meta.get("thr", self.best_thr))
            self.best_epochs = int(meta.get("epochs", self.best_epochs))

            self.scaler = joblib.load(self.scaler_file)

            m = BiggerMLP(in_dim=in_dim, out_dim=3)
            st = torch.load(self.model_file, map_location="cpu")
            m.load_state_dict(st)
            m.eval()
            self.model = m.to(self.device)

            self.status.emit(f"Modello caricato: {self.model_file} [{device_label(self.device)}]")
            self.report.emit(meta.get("last_report", ""))
            return True
        return False

    def _live_feature_vector(self, df_p: pd.DataFrame, df_s: pd.DataFrame) -> np.ndarray:
        feat = merge_primary_secondary(df_p, df_s, tf_s_ms=timeframe_to_ms(self.secondary_tf))
        row = feat.iloc[-1].copy()

        try:
            ob = self.ex_spot.fetch_order_book(self.symbol, limit=50)
            obf = orderbook_features(ob, depth=20)
        except Exception:
            obf = {"ob_spread": 0.0, "ob_top_imb": 0.0, "ob_depth_imb": 0.0, "ob_bid_depth": 0.0, "ob_ask_depth": 0.0}

        fut = try_fetch_futures_metrics(self.ex_fut)

        row["ob_spread"] = obf["ob_spread"]
        row["ob_top_imb"] = obf["ob_top_imb"]
        row["ob_depth_imb"] = obf["ob_depth_imb"]
        row["ob_bid_depth"] = obf["ob_bid_depth"]
        row["ob_ask_depth"] = obf["ob_ask_depth"]
        row["funding_rate"] = fut["funding_rate"]
        row["open_interest"] = fut["open_interest"]

        row = clamp_df(row.to_frame().T).iloc[0]
        return row.values.astype(np.float32)

    # -------------------------
    # AUTO-TUNING UNTIL >= 90%
    # -------------------------
    def evaluate_config(self, df_p, df_s, thr, horizon, epochs, conf_thr):
        X, y, used_df = make_dataset_multitf_3(
            df_p=df_p, df_s=df_s,
            horizon=int(horizon),
            thr=float(thr),
            tf_s_ms=timeframe_to_ms(self.secondary_tf)
        )
        if len(X) < 4000:
            return None

        (Xtr, ytr), (Xva, yva), (Xte, yte), (n_train, n_val) = chrono_split(X, y, self.cfg)

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
        Xva_s = scaler.transform(Xva).astype(np.float32)
        Xte_s = scaler.transform(Xte).astype(np.float32)

        model, best_val = train_one(
            Xtr_s, ytr, Xva_s, yva,
            device=self.device,
            epochs=int(epochs),
            lr=1e-3,
            batch_size=512
        )

        close_used = used_df["close"].astype(float).values
        te_start = n_train + n_val

        p_test = probs(model, Xte_s, device=self.device)
        bt_test = backtest_3class(close_used[te_start:], p_test, horizon=int(horizon), conf_thr=float(conf_thr), fee_bps=2.0)

        X_all_s = scaler.transform(X).astype(np.float32)
        wf = walk_forward_oos_3class(
            X_all_s, y, close_used, self.cfg,
            device=self.device,
            conf_thr=float(conf_thr),
            horizon=int(horizon),
            epochs=int(epochs)
        )

        return {
            "model": model,
            "scaler": scaler,
            "best_val": float(best_val),
            "bt_test": bt_test,
            "wf": wf,
            "thr": float(thr),
            "horizon": int(horizon),
            "epochs": int(epochs),
            "conf_thr": float(conf_thr),
            "in_dim": int(X.shape[1]),
        }

    def auto_search_until_90(self, df_p, df_s):
        thr_grid = self.cfg.get("thr_grid", [self.cfg["thr"]])
        horizon_grid = self.cfg.get("horizon_grid", [self.cfg["horizon_bars"]])
        epoch_grid = self.cfg.get("epoch_grid", [self.cfg["epochs"]])
        conf_grid = self.cfg.get("conf_grid", [0.55])

        best = None
        best_score = -1.0

        # Order chosen to increase NO-TRADE behavior gradually (higher conf & higher thr)
        for conf_thr in conf_grid[::-1]:
            for thr in thr_grid[::-1]:
                for horizon in horizon_grid:
                    for ep in epoch_grid:
                        if self._stop:
                            return None

                        self.status.emit(f"Tuning: conf={conf_thr:.2f} thr={thr} horizon={horizon} ep={ep}")
                        res = self.evaluate_config(df_p, df_s, thr=thr, horizon=horizon, epochs=ep, conf_thr=conf_thr)
                        if res is None:
                            continue

                        test_wr = res["bt_test"]["winrate"]
                        walk_wr = res["wf"]["winrate"]
                        test_tr = res["bt_test"]["trades"]
                        walk_tr = res["wf"]["trades"]

                        self.report.emit(
                            f"TUNE | conf={conf_thr:.2f} thr={thr} h={horizon} ep={ep} | "
                            f"TEST={test_wr*100:.1f}% ({test_tr} trades) | "
                            f"WALK={walk_wr*100:.1f}% ({walk_tr} trades)"
                        )

                        score = min(test_wr, walk_wr)
                        if score > best_score:
                            best_score = score
                            best = res

                        # PASS condition: >=90% with enough trades
                        if (test_wr >= MIN_WINRATE and walk_wr >= MIN_WINRATE and
                            test_tr >= MIN_TRADES_TEST and walk_tr >= MIN_TRADES_WALK):
                            self.status.emit("✅ Raggiunto vincolo 90% (TEST + WALK) con trade sufficienti.")
                            return res

        return best

    def _train_and_save(self):
        self.status.emit(f"Scarico storico KuCoin (molti dati)… [{self.primary_tf}/{self.secondary_tf}]")
        df_p = fetch_ohlcv_full(self.ex_spot, self.symbol, self.primary_tf, days=int(self.cfg["days_primary"]))
        df_s = fetch_ohlcv_full(self.ex_spot, self.symbol, self.secondary_tf, days=int(self.cfg["days_secondary"]))

        if len(df_p) < 2500 or len(df_s) < 1200:
            raise RuntimeError(f"Storico insufficiente: primary={len(df_p)} secondary={len(df_s)}")

        self.status.emit("Auto-tuning: continuo finché supero 90% (TEST + WALK) oppure fino al best possibile…")
        best = self.auto_search_until_90(df_p, df_s)
        if best is None:
            raise RuntimeError("Interrotto (ARRESTA) durante auto-tuning.")

        bt_test = best["bt_test"]
        wf = best["wf"]

        report = (
            f"PROFILE={self.profile_name} | DEV={device_label(self.device)} | TF={self.primary_tf}/{self.secondary_tf} | "
            f"conf={best['conf_thr']:.2f} thr={best['thr']} horizon={best['horizon']} epochs={best['epochs']} | "
            f"TEST winrate={bt_test['winrate']*100:.1f}% trades={bt_test['trades']} total={bt_test['total']*100:.1f}% | "
            f"WALK winrate={wf['winrate']*100:.1f}% trades={wf['trades']} seg={wf['segments']} total={wf['total']*100:.1f}% | "
            f"NOTE: orderbook/funding storici=0 (live-only)"
        )
        self.report.emit(report)

        # Must meet your rule to go LIVE
        if not (bt_test["winrate"] >= MIN_WINRATE and wf["winrate"] >= MIN_WINRATE and
                bt_test["trades"] >= MIN_TRADES_TEST and wf["trades"] >= MIN_TRADES_WALK):
            raise RuntimeError(
                f"Non raggiunto vincolo LIVE 90% con trade sufficienti. "
                f"Best trovato: TEST={bt_test['winrate']*100:.1f}% ({bt_test['trades']} trades) | "
                f"WALK={wf['winrate']*100:.1f}% ({wf['trades']} trades). "
                f"Per alzare ancora: aumenta conf_grid max o thr_grid max (più NO TRADE)."
            )

        self.status.emit("Salvo modello/scaler/meta…")
        torch.save({k: v.detach().cpu() for k, v in best["model"].state_dict().items()}, self.model_file)
        joblib.dump(best["scaler"], self.scaler_file)
        joblib.dump({
            "in_dim": best["in_dim"],
            "last_report": report,
            "conf_thr": best["conf_thr"],
            "thr": best["thr"],
            "horizon": best["horizon"],
            "epochs": best["epochs"],
        }, self.meta_file)

        self.model = best["model"]
        self.scaler = best["scaler"]
        self.best_conf_thr = best["conf_thr"]
        self.best_thr = best["thr"]
        self.best_horizon = best["horizon"]
        self.best_epochs = best["epochs"]

    def _ensure_model(self):
        if self._load_if_exists():
            return
        self.status.emit("Modello non trovato → AUTO-TRAIN + AUTO-TUNING…")
        self._train_and_save()
        self.status.emit("Training OK (>=90%) → pronto per realtime.")

    @QtCore.Slot()
    def run(self):
        try:
            self._ensure_model()
        except Exception as e:
            self.status.emit(f"ERRORE setup: {e.__class__.__name__}: {e}")
            self.stopped.emit()
            return

        self.ready.emit(self.model)
        self.status.emit("Realtime avviato…")

        while not self._stop:
            try:
                raw_p = self.ex_spot.fetch_ohlcv(self.symbol, timeframe=self.primary_tf, limit=self.window_live)
                raw_s = self.ex_spot.fetch_ohlcv(self.symbol, timeframe=self.secondary_tf, limit=self.window_live)

                df_p = pd.DataFrame(raw_p, columns=["timestamp","open","high","low","close","volume"])
                df_s = pd.DataFrame(raw_s, columns=["timestamp","open","high","low","close","volume"])

                feat_vec = self._live_feature_vector(df_p, df_s)

                X = feat_vec.reshape(1, -1)
                Xs = self.scaler.transform(X).astype(np.float32)

                with torch.no_grad():
                    Xt = torch.from_numpy(Xs).to(self.device)
                    logits = self.model(Xt)
                    pr = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

                p_long, p_short, p_nt = float(pr[0]), float(pr[1]), float(pr[2])

                # decision with 3-class + confidence gating
                cls = int(np.argmax([p_long, p_short, p_nt]))
                if cls == 2:
                    signal = "NO TRADE"
                else:
                    if max(p_long, p_short) < self.best_conf_thr:
                        signal = "NO TRADE"
                    else:
                        signal = "LONG" if cls == 0 else "SHORT"

                price = float(df_p["close"].iloc[-1])

                tp_pct = 0.0
                tp_price = 0.0
                if signal in ("LONG", "SHORT"):
                    atrp = atr_pct(df_p, period=14)
                    tp_pct, tp_price = suggest_takeprofit(
                        last_price=price,
                        atr_percent=atrp,
                        p_long=p_long,
                        p_short=p_short,
                        tp_base_mult=float(self.cfg["tp_base_mult"])
                    )

                self.tick.emit(Tick(time.time(), price, p_long, p_short, p_nt, signal, self.best_conf_thr, tp_pct, tp_price, feat_vec))

            except Exception as e:
                self.status.emit(f"ERRORE realtime: {e.__class__.__name__}: {e}")

            time.sleep(self.poll_s)

        self.status.emit("Sistema arrestato.")
        self.stopped.emit()


# =========================
# GUI (profile + start + stop + device popup)
# =========================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ETH KuCoin AI (LONG/SHORT/NO TRADE + auto-tuning >=90% + TP + GPU/CPU + STOP)")
        self.resize(1480, 860)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QGridLayout(central)

        # Controls
        self.profile = QtWidgets.QComboBox()
        self.profile.addItems(list(PROFILES.keys()))
        self.btn_start = QtWidgets.QPushButton("AVVIA")
        self.btn_stop = QtWidgets.QPushButton("ARRESTA")
        self.btn_stop.setEnabled(False)

        topbar = QtWidgets.QHBoxLayout()
        topbar.addWidget(QtWidgets.QLabel("Profilo"))
        topbar.addWidget(self.profile)
        topbar.addStretch(1)
        topbar.addWidget(self.btn_start)
        topbar.addWidget(self.btn_stop)

        # Plots
        self.price_plot = pg.PlotWidget(title="ETH Price (live)")
        self.prob_plot = pg.PlotWidget(title="Probabilities (LONG/SHORT/NO TRADE)")
        self.price_curve = self.price_plot.plot()
        self.long_curve = self.prob_plot.plot(name="LONG")
        self.short_curve = self.prob_plot.plot(name="SHORT")
        self.nt_curve = self.prob_plot.plot(name="NO TRADE")
        self.prob_plot.setYRange(0, 1)

        # Labels
        self.lbl_sig = QtWidgets.QLabel("Signal: -")
        self.lbl_sig.setStyleSheet("font-size: 24px; font-weight: 800;")
        self.lbl_price = QtWidgets.QLabel("Price: -")
        self.lbl_probs = QtWidgets.QLabel("P(LONG/SHORT/NO): - / - / -")
        self.lbl_conf = QtWidgets.QLabel("Confidence threshold: -")
        self.lbl_tp = QtWidgets.QLabel("Take Profit: -")
        self.lbl_device = QtWidgets.QLabel("Device: -")
        self.lbl_status = QtWidgets.QLabel("Status: pronto")
        self.lbl_report = QtWidgets.QLabel("Report: -")
        self.lbl_report.setWordWrap(True)

        info = QtWidgets.QVBoxLayout()
        info.addWidget(self.lbl_sig)
        info.addWidget(self.lbl_price)
        info.addWidget(self.lbl_probs)
        info.addWidget(self.lbl_conf)
        info.addWidget(self.lbl_tp)
        info.addWidget(self.lbl_device)
        info.addStretch(1)
        info.addWidget(self.lbl_report)
        info.addWidget(self.lbl_status)

        # 3D net
        self.model = BiggerMLP(in_dim=32, out_dim=3)
        self.net3d = Net3D(self.model, max_nodes_per_layer=48)

        layout.addLayout(topbar, 0, 0, 1, 4)
        layout.addWidget(self.price_plot, 1, 0, 1, 2)
        layout.addWidget(self.prob_plot, 2, 0, 1, 2)
        layout.addLayout(info, 1, 2, 2, 1)
        layout.addWidget(self.net3d, 1, 3, 2, 1)

        # buffers
        self.max_points = 300
        self.prices: Deque[float] = deque(maxlen=self.max_points)
        self.longps: Deque[float] = deque(maxlen=self.max_points)
        self.shortps: Deque[float] = deque(maxlen=self.max_points)
        self.ntps: Deque[float] = deque(maxlen=self.max_points)

        # worker thread
        self.thread: Optional[QtCore.QThread] = None
        self.worker: Optional[Worker] = None
        self.current_device: Optional[torch.device] = None

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

    def pick_device_popup(self) -> torch.device:
        options = ["CPU"]
        if has_cuda():
            options.append("GPU NVIDIA (CUDA)")

        item, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Seleziona Device",
            "Vuoi usare CPU o GPU NVIDIA?",
            options,
            0,
            False
        )
        if not ok:
            return torch.device("cpu")

        if item.startswith("GPU") and has_cuda():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def start(self):
        prof = self.profile.currentText().strip()
        dev = self.pick_device_popup()
        self.current_device = dev
        self.lbl_device.setText(f"Device: {device_label(dev)}")

        self.btn_start.setEnabled(False)
        self.profile.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Status: avvio… (auto-tuning fino a >=90%)")

        self.thread = QtCore.QThread()
        self.worker = Worker(profile_name=prof, device=dev)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.on_status)
        self.worker.report.connect(self.on_report)
        self.worker.tick.connect(self.on_tick)
        self.worker.ready.connect(self.on_ready)
        self.worker.stopped.connect(self.on_stopped)

        self.thread.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
        if self.thread:
            self.thread.quit()
            self.thread.wait(2500)
        self.on_stopped()

    @QtCore.Slot()
    def on_stopped(self):
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.profile.setEnabled(True)

    @QtCore.Slot(str)
    def on_status(self, msg: str):
        self.lbl_status.setText(f"Status: {msg}")
        if msg.startswith("ERRORE setup"):
            self.on_stopped()

    @QtCore.Slot(str)
    def on_report(self, msg: str):
        self.lbl_report.setText(f"Report: {msg}")

    @QtCore.Slot(object)
    def on_ready(self, model: BiggerMLP):
        self.model = model
        self.net3d.set_model(self.model)

    @QtCore.Slot(object)
    def on_tick(self, t: Tick):
        self.prices.append(t.price)
        self.longps.append(t.long_p)
        self.shortps.append(t.short_p)
        self.ntps.append(t.nt_p)

        x = np.arange(len(self.prices))
        self.price_curve.setData(x, np.array(self.prices, dtype=np.float32))
        self.long_curve.setData(x, np.array(self.longps, dtype=np.float32))
        self.short_curve.setData(x, np.array(self.shortps, dtype=np.float32))
        self.nt_curve.setData(x, np.array(self.ntps, dtype=np.float32))

        self.lbl_sig.setText(f"Signal: {t.signal}")
        self.lbl_price.setText(f"Price: {t.price:,.2f}")
        self.lbl_probs.setText(f"P(LONG/SHORT/NO): {t.long_p:.3f} / {t.short_p:.3f} / {t.nt_p:.3f}")
        self.lbl_conf.setText(f"Confidence threshold: {t.conf_thr:.2f}")

        if t.signal in ("LONG", "SHORT"):
            target = (t.price + t.tp_price) if t.signal == "LONG" else (t.price - t.tp_price)
            self.lbl_tp.setText(f"Take Profit: {t.tp_pct*100:.2f}% | Δ {t.tp_price:,.2f} | Target {target:,.2f}")
        else:
            self.lbl_tp.setText("Take Profit: -")

        self.net3d.update_from_cache(t.feat, t.long_p, t.short_p, t.nt_p)

    def closeEvent(self, event):
        try:
            self.stop()
        finally:
            event.accept()


def main():
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
