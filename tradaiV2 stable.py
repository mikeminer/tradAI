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
MIN_WINRATE = 0.90  # 90% required to go live (TEST and Walk-forward)


# =========================
# CONFIG: Profiles (few UI commands)
# More training data than before
# =========================
PROFILES = {
    "SCALPING (1m/3m)": {
        "primary_tf": "1m",
        "secondary_tf": "3m",
        "poll_s": 2.0,
        # more data:
        "days_primary": 45,     # was 14
        "days_secondary": 90,   # was 30
        "horizon_bars": 6,
        "thr": 0.0012,
        "train_window_frac": 0.60,
        "val_window_frac": 0.20,
        "test_window_frac": 0.20,
        "walk_train_bars": 3500,  # more
        "walk_test_bars": 500,
        "epochs": 12,
    },
    "SWING (15m/1h)": {
        "primary_tf": "15m",
        "secondary_tf": "1h",
        "poll_s": 5.0,
        # more data:
        "days_primary": 540,    # was 240
        "days_secondary": 900,  # was 365
        "horizon_bars": 8,
        "thr": 0.0025,
        "train_window_frac": 0.60,
        "val_window_frac": 0.20,
        "test_window_frac": 0.20,
        "walk_train_bars": 3500,
        "walk_test_bars": 400,
        "epochs": 14,
    }
}

SYMBOL_SPOT = "ETH/USDT"   # KuCoin spot
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
        return {
            "ob_spread": 0.0,
            "ob_top_imb": 0.0,
            "ob_depth_imb": 0.0,
            "ob_bid_depth": 0.0,
            "ob_ask_depth": 0.0
        }

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
            rate = fr.get("fundingRate", fr.get("funding_rate", 0.0))
            out["funding_rate"] = float(rate or 0.0)
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
# MULTI-TF DATASET BUILD
# =========================
def merge_primary_secondary(df_p: pd.DataFrame, df_s: pd.DataFrame, tf_s_ms: int) -> pd.DataFrame:
    fp = features_ohlcv(df_p, prefix="p_")
    fs = features_ohlcv(df_s, prefix="s_")

    p = df_p[["timestamp"]].copy()
    s = df_s[["timestamp"]].copy()
    p["ts"] = pd.to_datetime(p["timestamp"], unit="ms")
    s["ts"] = pd.to_datetime(s["timestamp"], unit="ms")

    fp2 = fp.copy()
    fs2 = fs.copy()
    fp2["ts"] = p["ts"].values
    fs2["ts"] = s["ts"].values

    fp2 = fp2.sort_values("ts")
    fs2 = fs2.sort_values("ts")

    merged = pd.merge_asof(fp2, fs2, on="ts", direction="backward", tolerance=pd.Timedelta(milliseconds=tf_s_ms*2))
    merged = merged.drop(columns=["ts"]).fillna(0)
    merged = clamp_df(merged)
    return merged


# =========================
# LABELING
# =========================
def build_labels(close: np.ndarray, horizon: int, thr: float) -> np.ndarray:
    y = np.full(len(close), -1, dtype=np.int64)
    for i in range(len(close) - horizon):
        now = close[i]
        future = close[i + horizon]
        if future > now * (1 + thr):
            y[i] = 0  # LONG
        elif future < now * (1 - thr):
            y[i] = 1  # SHORT
    return y

def make_dataset_multitf(df_p: pd.DataFrame, df_s: pd.DataFrame, horizon: int, thr: float, tf_s_ms: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    feat = merge_primary_secondary(df_p, df_s, tf_s_ms=tf_s_ms)

    # placeholders for live-only features (training deterministic, not random)
    for col in ["ob_spread", "ob_top_imb", "ob_depth_imb", "ob_bid_depth", "ob_ask_depth", "funding_rate", "open_interest"]:
        feat[col] = 0.0

    close = df_p["close"].astype(float).values
    y = build_labels(close, horizon=horizon, thr=thr)

    n = min(len(feat), len(y))
    feat = feat.iloc[:n].reset_index(drop=True)
    df_p_used = df_p.iloc[:n].reset_index(drop=True)
    y = y[:n]

    mask = (y != -1)
    X = feat.values.astype(np.float32)[mask]
    Y = y[mask]
    used_df = df_p_used.loc[mask].reset_index(drop=True)
    used_feat = feat.loc[mask].reset_index(drop=True)
    return X, Y, used_df, used_feat


# =========================
# MODEL
# =========================
class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, h1=64, h2=32, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)
        self.last_a1 = None
        self.last_a2 = None
        self.last_out = None

    def forward(self, x):
        a1 = torch.tanh(self.fc1(x))
        a2 = torch.tanh(self.fc2(a1))
        out = self.fc3(a2)
        self.last_a1 = a1.detach().cpu().numpy()
        self.last_a2 = a2.detach().cpu().numpy()
        self.last_out = out.detach().cpu().numpy()
        return out


# =========================
# TRAIN / EVAL
# =========================
def chrono_split(X: np.ndarray, y: np.ndarray, cfg: dict):
    n = len(X)
    n_train = int(n * cfg["train_window_frac"])
    n_val = int(n * cfg["val_window_frac"])
    n_test = n - n_train - n_val
    if n_train < 300 or n_val < 200 or n_test < 200:
        raise RuntimeError(f"Dataset too small for split: n={n} (train={n_train}, val={n_val}, test={n_test})")

    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    Xte, yte = X[n_train+n_val:], y[n_train+n_val:]
    return (Xtr, ytr), (Xva, yva), (Xte, yte), (n_train, n_val)

def train_one(Xtr, ytr, Xva, yva, epochs: int = 10, lr: float = 1e-3, batch_size: int = 256):
    in_dim = Xtr.shape[1]
    model = TinyMLP(in_dim=in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr)
    Xv = torch.from_numpy(Xva)
    yv = torch.from_numpy(yva)

    best_loss = float("inf")
    best_state = None

    for _ep in range(epochs):
        model.train()
        n = len(Xt)
        idx = torch.randperm(n)
        Xt_sh = Xt[idx]
        yt_sh = yt[idx]

        for start in range(0, n, batch_size):
            xb = Xt_sh[start:start+batch_size]
            yb = yt_sh[start:start+batch_size]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = float(F.cross_entropy(val_logits, yv).cpu().item())

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_loss

def probs(model: TinyMLP, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        return F.softmax(logits, dim=1).cpu().numpy()

def backtest_fixed_horizon(close: np.ndarray, probs2: np.ndarray, horizon: int, fee_bps: float = 2.0) -> dict:
    n = min(len(probs2), len(close) - horizon)
    if n <= 0:
        return {"trades": 0, "winrate": 0.0, "avg_ret": 0.0, "total": 0.0}

    fee = (fee_bps / 10_000.0)
    rets = []
    for i in range(n):
        entry = close[i]
        exitp = close[i + horizon]
        p_long, p_short = probs2[i]

        if p_long >= p_short:
            r = (exitp / entry) - 1.0
        else:
            r = (entry / exitp) - 1.0

        r = r - 2 * fee
        rets.append(r)

    rets = np.array(rets, dtype=np.float64)
    trades = len(rets)
    winrate = float((rets > 0).mean())
    avg_ret = float(rets.mean())
    total = float(np.prod(1.0 + rets) - 1.0)
    return {"trades": trades, "winrate": winrate, "avg_ret": avg_ret, "total": total}

def walk_forward_oos(X: np.ndarray, y: np.ndarray, close: np.ndarray, cfg: dict) -> dict:
    train_w = int(cfg["walk_train_bars"])
    test_w = int(cfg["walk_test_bars"])
    epochs = int(cfg["epochs"])

    if len(X) < train_w + test_w + 500:
        return {"segments": 0, "trades": 0, "winrate": 0.0, "avg_ret": 0.0, "total": 0.0}

    seg_stats = []
    start = 0
    while True:
        tr_start = start
        tr_end = tr_start + train_w
        te_start = tr_end
        te_end = te_start + test_w
        if te_end > len(X):
            break

        Xtr, ytr = X[tr_start:tr_end], y[tr_start:tr_end]
        cut = int(len(Xtr) * 0.85)
        Xtr2, ytr2 = Xtr[:cut], ytr[:cut]
        Xva2, yva2 = Xtr[cut:], ytr[cut:]

        model, _ = train_one(Xtr2, ytr2, Xva2, yva2, epochs=epochs, lr=1e-3, batch_size=256)
        p = probs(model, X[te_start:te_end])

        cseg = close[te_start:te_end]
        bt = backtest_fixed_horizon(cseg, p, horizon=int(cfg["horizon_bars"]), fee_bps=2.0)
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
    return {"segments": len(seg_stats), "trades": trades, "winrate": float(winrate), "avg_ret": float(avg_ret), "total": float(total)}


# =========================
# 3D NET VIEW (LONG green, SHORT red)
# =========================
class Net3D(gl.GLViewWidget):
    def __init__(self, model: TinyMLP):
        super().__init__()
        self.model = model
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

    def _rebuild(self):
        for it in list(self.items):
            if not isinstance(it, gl.GLGridItem):
                self.removeItem(it)

        in_dim = self.model.fc1.in_features
        h1 = self.model.fc1.out_features
        h2 = self.model.fc2.out_features
        out_dim = 2  # LONG/SHORT

        self.pos_in = self._layer_positions(in_dim, x=-4.0)
        self.pos_h1 = self._layer_positions(h1, x=-1.0)
        self.pos_h2 = self._layer_positions(h2, x=2.0)

        # output: 2 nodes fixed positions
        self.pos_out = np.array([[5.0, 1.0, 0.0], [5.0, -1.0, 0.0]], dtype=np.float32)

        self.s_in = gl.GLScatterPlotItem(pos=self.pos_in, size=6)
        self.s_h1 = gl.GLScatterPlotItem(pos=self.pos_h1, size=8)
        self.s_h2 = gl.GLScatterPlotItem(pos=self.pos_h2, size=8)
        self.s_out = gl.GLScatterPlotItem(pos=self.pos_out, size=12)

        self.addItem(self.s_in); self.addItem(self.s_h1); self.addItem(self.s_h2); self.addItem(self.s_out)

        self.e1 = self._make_edges(self.pos_in, self.pos_h1)
        self.e2 = self._make_edges(self.pos_h1, self.pos_h2)
        self.e3 = self._make_edges(self.pos_h2, self.pos_out)
        self.addItem(self.e1); self.addItem(self.e2); self.addItem(self.e3)

        self._set_gray_intensity(self.s_in, np.zeros(in_dim))
        self._set_gray_intensity(self.s_h1, np.zeros(h1))
        self._set_gray_intensity(self.s_h2, np.zeros(h2))

        # Output colors start dim
        self._set_output_colors(0.5, 0.5)

    def _set_output_colors(self, p_long: float, p_short: float):
        # LONG green, SHORT red. Intensity follows probability.
        pl = float(np.clip(p_long, 0.0, 1.0))
        ps = float(np.clip(p_short, 0.0, 1.0))

        # LONG node: green channel intensity
        c_long = np.array([0.0, pl, 0.0, 1.0], dtype=np.float32)
        # SHORT node: red channel intensity
        c_short = np.array([ps, 0.0, 0.0, 1.0], dtype=np.float32)

        colors = np.vstack([c_long, c_short]).astype(np.float32)
        self.s_out.setData(color=colors)

    def set_model(self, model: TinyMLP):
        self.model = model
        self._rebuild()

    def update_from_cache(self, feat_vec: np.ndarray, p_long: float, p_short: float):
        inp = np.tanh(np.abs(feat_vec))
        self._set_gray_intensity(self.s_in, inp)

        if self.model.last_a1 is not None:
            self._set_gray_intensity(self.s_h1, self.model.last_a1[0])
        if self.model.last_a2 is not None:
            self._set_gray_intensity(self.s_h2, self.model.last_a2[0])

        self._set_output_colors(p_long, p_short)


# =========================
# WORKER
# =========================
@dataclass
class Tick:
    ts: float
    price: float
    long_p: float
    short_p: float
    signal: str
    feat: np.ndarray

class Worker(QtCore.QObject):
    status = QtCore.Signal(str)
    report = QtCore.Signal(str)
    tick = QtCore.Signal(object)
    ready = QtCore.Signal(object)  # model
    stopped = QtCore.Signal()

    def __init__(self, profile_name: str):
        super().__init__()
        self.cfg = PROFILES[profile_name]
        self.profile_name = profile_name
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

        tag = safe_name(profile_name)
        self.model_file = f"model_{tag}.pt"
        self.scaler_file = f"scaler_{tag}.joblib"
        self.meta_file = f"meta_{tag}.joblib"

        self.model: Optional[TinyMLP] = None
        self.scaler: Optional[StandardScaler] = None

        self.window_live = 300

    def stop(self):
        self._stop = True

    def _load_if_exists(self) -> bool:
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file) and os.path.exists(self.meta_file):
            meta = joblib.load(self.meta_file)
            in_dim = int(meta["in_dim"])
            self.scaler = joblib.load(self.scaler_file)

            m = TinyMLP(in_dim=in_dim)
            st = torch.load(self.model_file, map_location="cpu")
            m.load_state_dict(st)
            m.eval()
            self.model = m

            self.status.emit(f"Modello caricato: {self.model_file}")
            self.report.emit(meta.get("last_report", ""))
            return True
        return False

    def _train_and_save(self):
        self.status.emit("Scarico storico KuCoin (più dati)…")
        df_p = fetch_ohlcv_full(self.ex_spot, self.symbol, self.primary_tf, days=int(self.cfg["days_primary"]))
        df_s = fetch_ohlcv_full(self.ex_spot, self.symbol, self.secondary_tf, days=int(self.cfg["days_secondary"]))

        if len(df_p) < 1200 or len(df_s) < 600:
            raise RuntimeError(f"Storico insufficiente: primary={len(df_p)} secondary={len(df_s)}")

        self.status.emit("Creo dataset multi-timeframe…")
        X, y, used_df, used_feat = make_dataset_multitf(
            df_p=df_p, df_s=df_s,
            horizon=int(self.cfg["horizon_bars"]),
            thr=float(self.cfg["thr"]),
            tf_s_ms=timeframe_to_ms(self.secondary_tf)
        )
        if len(X) < 2000:
            raise RuntimeError(f"Dataset troppo piccolo dopo filtro neutral: {len(X)} esempi.")

        (Xtr, ytr), (Xva, yva), (Xte, yte), (n_train, n_val) = chrono_split(X, y, self.cfg)

        self.status.emit("Normalizzo (fit solo su TRAIN)…")
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
        Xva_s = scaler.transform(Xva).astype(np.float32)
        Xte_s = scaler.transform(Xte).astype(np.float32)

        self.status.emit("Training con best model su VALIDATION…")
        model, best_val = train_one(Xtr_s, ytr, Xva_s, yva, epochs=int(self.cfg["epochs"]), lr=1e-3, batch_size=256)

        self.status.emit("Valuto TEST (out-of-sample)…")
        close_used = used_df["close"].astype(float).values
        te_start = n_train + n_val

        p_test = probs(model, Xte_s)
        bt_test = backtest_fixed_horizon(close_used[te_start:], p_test, horizon=int(self.cfg["horizon_bars"]), fee_bps=2.0)

        self.status.emit("Walk-forward automatico OOS…")
        X_all_s = scaler.transform(X).astype(np.float32)
        wf = walk_forward_oos(X_all_s, y, close_used, self.cfg)

        report_lines = [
            f"PROFILE={self.profile_name}",
            f"TF={self.primary_tf}/{self.secondary_tf}",
            f"horizon={self.cfg['horizon_bars']} thr={self.cfg['thr']}",
            f"best_val_loss={best_val:.4f}",
            f"TEST winrate={bt_test['winrate']*100:.1f}% trades={bt_test['trades']} total={bt_test['total']*100:.1f}%",
            f"WALK OOS winrate={wf['winrate']*100:.1f}% trades={wf['trades']} segments={wf['segments']} total={wf['total']*100:.1f}%",
            "orderbook/funding storici=0 (live-only)"
        ]
        report = " | ".join(report_lines)
        self.report.emit(report)

        # HARD GATE: require >= 90% both
        if bt_test["winrate"] < MIN_WINRATE or wf["winrate"] < MIN_WINRATE:
            raise RuntimeError(
                f"Winrate insufficiente per andare LIVE. "
                f"Richiesto >= {MIN_WINRATE*100:.0f}%. "
                f"TEST={bt_test['winrate']*100:.1f}% | WALK={wf['winrate']*100:.1f}%"
            )

        self.status.emit("Salvo modello/scaler/meta…")
        tag = safe_name(self.profile_name)
        torch.save(model.state_dict(), self.model_file)
        joblib.dump(scaler, self.scaler_file)
        joblib.dump({"in_dim": X.shape[1], "last_report": report}, self.meta_file)

        self.model = model
        self.scaler = scaler

    def _ensure_model(self):
        if self._load_if_exists():
            return
        self.status.emit("Modello non trovato → AUTO-TRAIN…")
        self._train_and_save()
        self.status.emit("Training OK (winrate >= 90%) → pronto per realtime.")

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
                df_p = self.ex_spot.fetch_ohlcv(self.symbol, timeframe=self.primary_tf, limit=self.window_live)
                df_s = self.ex_spot.fetch_ohlcv(self.symbol, timeframe=self.secondary_tf, limit=self.window_live)

                df_p = pd.DataFrame(df_p, columns=["timestamp","open","high","low","close","volume"])
                df_s = pd.DataFrame(df_s, columns=["timestamp","open","high","low","close","volume"])

                feat_vec = self._live_feature_vector(df_p, df_s)

                X = feat_vec.reshape(1, -1)
                Xs = self.scaler.transform(X).astype(np.float32)

                with torch.no_grad():
                    logits = self.model(torch.from_numpy(Xs))
                    pr = F.softmax(logits, dim=1).cpu().numpy()[0]

                long_p = float(pr[0])
                short_p = float(pr[1])
                signal = "LONG" if long_p >= short_p else "SHORT"
                price = float(df_p["close"].iloc[-1])

                self.tick.emit(Tick(time.time(), price, long_p, short_p, signal, feat_vec))

            except Exception as e:
                self.status.emit(f"ERRORE realtime: {e.__class__.__name__}: {e}")

            time.sleep(self.poll_s)

        self.status.emit("Sistema arrestato.")
        self.stopped.emit()


# =========================
# GUI (profile + start + stop)
# =========================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ETH KuCoin AI (>=90% winrate gate + more data + stop + colored outputs)")
        self.resize(1350, 780)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QGridLayout(central)

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

        self.price_plot = pg.PlotWidget(title="ETH Price (live)")
        self.prob_plot = pg.PlotWidget(title="Output Probabilities")
        self.price_curve = self.price_plot.plot()
        self.long_curve = self.prob_plot.plot(name="LONG")
        self.short_curve = self.prob_plot.plot(name="SHORT")
        self.prob_plot.setYRange(0, 1)

        self.lbl_sig = QtWidgets.QLabel("Signal: -")
        self.lbl_sig.setStyleSheet("font-size: 24px; font-weight: 800;")
        self.lbl_price = QtWidgets.QLabel("Price: -")
        self.lbl_probs = QtWidgets.QLabel("LONG/SHORT: - / -")
        self.lbl_status = QtWidgets.QLabel("Status: pronto")
        self.lbl_report = QtWidgets.QLabel("Report: -")
        self.lbl_report.setWordWrap(True)

        info = QtWidgets.QVBoxLayout()
        info.addWidget(self.lbl_sig)
        info.addWidget(self.lbl_price)
        info.addWidget(self.lbl_probs)
        info.addStretch(1)
        info.addWidget(self.lbl_report)
        info.addWidget(self.lbl_status)

        # placeholder model
        self.model = TinyMLP(in_dim=32)
        self.net3d = Net3D(self.model)

        layout.addLayout(topbar, 0, 0, 1, 4)
        layout.addWidget(self.price_plot, 1, 0, 1, 2)
        layout.addWidget(self.prob_plot, 2, 0, 1, 2)
        layout.addLayout(info, 1, 2, 2, 1)
        layout.addWidget(self.net3d, 1, 3, 2, 1)

        self.max_points = 300
        self.prices: Deque[float] = deque(maxlen=self.max_points)
        self.longps: Deque[float] = deque(maxlen=self.max_points)
        self.shortps: Deque[float] = deque(maxlen=self.max_points)

        self.thread: Optional[QtCore.QThread] = None
        self.worker: Optional[Worker] = None

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

    def start(self):
        prof = self.profile.currentText().strip()

        self.btn_start.setEnabled(False)
        self.profile.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.lbl_status.setText("Status: avvio… (training + gate winrate >= 90%)")

        self.thread = QtCore.QThread()
        self.worker = Worker(profile_name=prof)

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
            self.thread.wait(2000)

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
            # unlock UI if training gate fails
            self.on_stopped()

    @QtCore.Slot(str)
    def on_report(self, msg: str):
        self.lbl_report.setText(f"Report: {msg}")

    @QtCore.Slot(object)
    def on_ready(self, model: TinyMLP):
        self.model = model
        self.net3d.set_model(self.model)

    @QtCore.Slot(object)
    def on_tick(self, t: Tick):
        self.prices.append(t.price)
        self.longps.append(t.long_p)
        self.shortps.append(t.short_p)

        x = np.arange(len(self.prices))
        self.price_curve.setData(x, np.array(self.prices, dtype=np.float32))
        self.long_curve.setData(x, np.array(self.longps, dtype=np.float32))
        self.short_curve.setData(x, np.array(self.shortps, dtype=np.float32))

        self.lbl_sig.setText(f"Signal: {t.signal}")
        self.lbl_price.setText(f"Price: {t.price:,.2f}")
        self.lbl_probs.setText(f"LONG/SHORT: {t.long_p:.3f} / {t.short_p:.3f}")

        # long green, short red in 3D output nodes
        self.net3d.update_from_cache(t.feat, t.long_p, t.short_p)

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
