"""
Entorno de trading compatible con Gymnasium — v3 con indicadores + multi-timeframe.

Estado del agente (432 features):
  - 50 velas de 5m × 7 features = 350
      ret, h_rel, l_rel, o_rel, v_rel   (precio bruto normalizado)
      rsi_norm, bb_pct_norm             (indicadores tectnicos, nuevos)
  - 20 velas de 1h × 4 features = 80
      ret_1h, v_rel_1h, rsi_1h, bb_1h  (contexto de tendencia mayor)
  - 2 features de posicion
      pos_enc, pnl_latente

El RSI y las Bandas de Bollinger dan al agente señales mas directas
de sobrecompra/sobreventa sin eliminar el enfoque de RL puro —
la red sigue aprendiendo CUANDO y COMO usar esas señales.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import config
from env.indicators import rsi_norm, bb_pct_norm


class TradingEnv(gym.Env):

    FLAT  = 0
    LONG  = 1
    SHORT = 2

    WINDOW_1H = 20   # ultimas 20 velas de 1h como contexto de tendencia

    def __init__(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame = None,
                 window: int = config.WINDOW,
                 initial_capital: float = config.INITIAL_CAP):
        super().__init__()

        self.df              = df_5m.reset_index(drop=True)
        self.window          = window
        self.initial_capital = initial_capital

        # ── Pre-computar indicadores 5m ───────────────────────────────────────
        # Una sola vez al crear el entorno — O(N), no O(N) por step
        close_5m      = df_5m["close"]
        self._rsi_5m  = rsi_norm(close_5m)       # shape (N,)
        self._bb_5m   = bb_pct_norm(close_5m)    # shape (N,)

        # ── Pre-computar indicadores 1h ───────────────────────────────────────
        self.has_1h = df_1h is not None
        if self.has_1h:
            self.df_1h       = df_1h.reset_index(drop=True)
            close_1h         = df_1h["close"]
            self._rsi_1h     = rsi_norm(close_1h)
            self._bb_1h      = bb_pct_norm(close_1h)

            # Mapeado 5m → 1h: BTC opera 24/7, 1h = exactamente 12 velas de 5m
            # Para cada indice 5m, el indice 1h correspondiente es idx // 12
            n_1h = len(self.df_1h)
            self._1h_of      = np.clip(
                np.arange(len(self.df)) // 12,
                self.WINDOW_1H, n_1h - 1
            )
            # Pre-computar features 1h como array numpy (N_1h, 4) para acceso rapido
            c1h = self.df_1h["close"].values
            v1h = self.df_1h["volume"].values
            v1h_mean = (pd.Series(v1h).rolling(20, min_periods=1).mean().values + 1e-9)
            ret_1h = np.diff(c1h, prepend=c1h[0]) / (c1h + 1e-9)
            vr_1h  = v1h / v1h_mean
            self._feat_1h = np.column_stack([
                ret_1h,
                vr_1h,
                self._rsi_1h,
                self._bb_1h,
            ]).astype(np.float32)   # (N_1h, 4)

        # ── Espacio de observacion ────────────────────────────────────────────
        feat_5m = window * 7                              # 350
        feat_1h = self.WINDOW_1H * 4 if self.has_1h else 0  # 80
        self.state_size = feat_5m + feat_1h + 2          # 432

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opts       = options or {}
        start_idx  = opts.get("start_idx",    self.window)
        episode_len= opts.get("episode_len",  None)
        self._reset_state(start_idx=start_idx, episode_len=episode_len)
        return self._get_obs(), {}

    def step(self, action: int):
        price  = float(self.df["close"].iloc[self.idx])
        reward = 0.0

        if action == self.FLAT:
            if self.position != 0:
                reward += self._close(price)

        elif action == self.LONG:
            if self.position == -1:
                reward += self._close(price)
            if self.position != 1:
                self.position    = 1
                self.entry_price = price
                self.hold_steps  = 0

        elif action == self.SHORT:
            if self.position == 1:
                reward += self._close(price)
            if self.position != -1:
                self.position    = -1
                self.entry_price = price
                self.hold_steps  = 0

        # Penalizacion por drawdown grande
        if self.position != 0:
            self.hold_steps += 1
            unrealized = (price - self.entry_price) / self.entry_price * self.position
            if unrealized < -0.015:
                reward -= 0.003 * config.REWARD_SCALE
            if unrealized < -0.030:
                reward -= 0.005 * config.REWARD_SCALE

        self.idx        += 1
        self.step_count += 1
        done = self.idx >= self.end_idx

        if done and self.position != 0:
            price   = float(self.df["close"].iloc[-1])
            reward += self._close(price)

        return self._get_obs(), float(reward), done, False, {
            "position": self.position,
            "capital":  self.capital,
            "n_trades": self.n_trades,
        }

    # ── Internos ──────────────────────────────────────────────────────────────

    def refresh_live(self, df_5m: pd.DataFrame, df_1h: pd.DataFrame = None):
        """
        Actualiza los datos y re-calcula los indicadores para el tick en vivo.
        Llamar antes de _get_obs() cada vez que llegan datos frescos del exchange.
        """
        self.df    = df_5m.reset_index(drop=True)
        self.idx   = len(self.df) - 1

        close_5m       = df_5m["close"]
        self._rsi_5m   = rsi_norm(close_5m)
        self._bb_5m    = bb_pct_norm(close_5m)

        if df_1h is not None and self.has_1h:
            self.df_1h     = df_1h.reset_index(drop=True)
            close_1h       = df_1h["close"]
            self._rsi_1h   = rsi_norm(close_1h)
            self._bb_1h    = bb_pct_norm(close_1h)

            n_1h           = len(self.df_1h)
            self._1h_of    = np.clip(
                np.arange(len(self.df)) // 12,
                self.WINDOW_1H, n_1h - 1
            )
            c1h            = self.df_1h["close"].values
            v1h            = self.df_1h["volume"].values
            v1h_mean       = (pd.Series(v1h).rolling(20, min_periods=1).mean().values + 1e-9)
            ret_1h         = np.diff(c1h, prepend=c1h[0]) / (c1h + 1e-9)
            vr_1h          = v1h / v1h_mean
            self._feat_1h  = np.column_stack([
                ret_1h, vr_1h, self._rsi_1h, self._bb_1h,
            ]).astype(np.float32)

    def _reset_state(self, start_idx: int = None, episode_len: int = None):
        self.idx         = start_idx if start_idx is not None else self.window
        self.end_idx     = (self.idx + episode_len) if episode_len else len(self.df) - 1
        self.end_idx     = min(self.end_idx, len(self.df) - 1)
        self.position    = 0
        self.entry_price = 0.0
        self.capital     = self.initial_capital
        self.step_count  = 0
        self.n_trades    = 0
        self.hold_steps  = 0

    def _close(self, price: float) -> float:
        pnl_pct = (price - self.entry_price) / self.entry_price * self.position
        fee     = config.TRADE_FEE * 2
        net_pnl = pnl_pct - fee
        self.capital    *= (1 + net_pnl)
        self.position    = 0
        self.entry_price = 0.0
        self.hold_steps  = 0
        self.n_trades   += 1
        return float(net_pnl * config.REWARD_SCALE)

    def _get_obs(self) -> np.ndarray:
        w   = self.df.iloc[self.idx - self.window: self.idx]
        c   = w["close"].values
        eps = 1e-9

        # ── Features 5m ──────────────────────────────────────────────────────
        ret   = np.diff(c, prepend=c[0]) / (c + eps)
        h_rel = (w["high"].values  - c) / (c + eps)
        l_rel = (w["low"].values   - c) / (c + eps)
        o_rel = (w["open"].values  - c) / (c + eps)
        vol   = w["volume"].values
        v_rel = vol / (np.convolve(vol, np.ones(20)/20, mode="same") + eps)

        rsi_w = self._rsi_5m[self.idx - self.window: self.idx]
        bb_w  = self._bb_5m [self.idx - self.window: self.idx]

        features_5m = np.column_stack(
            [ret, h_rel, l_rel, o_rel, v_rel, rsi_w, bb_w]
        ).flatten()   # 350

        # ── Features 1h ──────────────────────────────────────────────────────
        if self.has_1h:
            idx_1h  = int(self._1h_of[self.idx])
            start_1h = max(0, idx_1h - self.WINDOW_1H)
            chunk   = self._feat_1h[start_1h: idx_1h]  # (≤20, 4)

            # Rellenar con ceros si hay menos de WINDOW_1H velas disponibles
            if len(chunk) < self.WINDOW_1H:
                pad = np.zeros((self.WINDOW_1H - len(chunk), 4), dtype=np.float32)
                chunk = np.vstack([pad, chunk])

            features_1h = chunk.flatten()   # 80
        else:
            features_1h = np.array([], dtype=np.float32)

        # ── Posicion ──────────────────────────────────────────────────────────
        price   = c[-1]
        pos_enc = float(self.position)
        pnl_lat = 0.0
        if self.position != 0 and self.entry_price > 0:
            pnl_lat = (price - self.entry_price) / self.entry_price * self.position

        obs = np.concatenate(
            [features_5m, features_1h, [pos_enc, pnl_lat]]
        ).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(obs, -10.0, 10.0)
