"""
Entorno de trading compatible con Gymnasium.

El agente ve las ultimas WINDOW velas normalizadas y decide:
  0 = Hold     (no hacer nada)
  1 = Long     (comprar / mantener long)
  2 = Short    (vender / mantener short)
  3 = Close    (cerrar posicion actual)

No le decimos que mire RSI, EMA ni ningun indicador.
Solo recibe precios brutos normalizados — aprende el solo que patrones importan.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import config


class TradingEnv(gym.Env):
    """
    Parametros
    ----------
    df : DataFrame con columnas [open, high, low, close, volume]
    window : numero de velas pasadas que ve el agente como estado
    """

    HOLD  = 0
    LONG  = 1
    SHORT = 2
    CLOSE = 3

    def __init__(self, df: pd.DataFrame, window: int = config.WINDOW,
                 initial_capital: float = config.INITIAL_CAP):
        super().__init__()
        self.df             = df.reset_index(drop=True)
        self.window         = window
        self.initial_capital = initial_capital

        # Estado: window velas * 5 features (ret, h/c, l/c, o/c, vol_ratio)
        #         + 2 valores de posicion (pos_enc, pnl_unrealizado)
        self.state_size = window * 5 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self._reset_state()

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        price  = float(self.df["close"].iloc[self.idx])
        reward = 0.0

        # ── Ejecutar accion ───────────────────────────────────────────────
        if action == self.LONG:
            if self.position == -1:               # cierra short
                reward += self._close(price)
            if self.position != 1:
                self.position    = 1
                self.entry_price = price

        elif action == self.SHORT:
            if self.position == 1:                # cierra long
                reward += self._close(price)
            if self.position != -1:
                self.position    = -1
                self.entry_price = price

        elif action == self.CLOSE:
            if self.position != 0:
                reward += self._close(price)

        # else HOLD: no hacer nada

        # Coste de oportunidad: penalizar ligeramente por cada tick sin hacer nada
        # Incentiva al agente a buscar operaciones activamente
        if self.position == 0:
            reward -= 0.0001

        # Penalizacion por mantener una posicion perdedora demasiado tiempo
        if self.position != 0:
            unrealized = (price - self.entry_price) / self.entry_price * self.position
            if unrealized < -0.015:              # perdida > 1.5% → penalizar mas
                reward -= 0.001

        self.idx       += 1
        self.step_count += 1
        done = self.idx >= len(self.df) - 1

        # Forzar cierre al final del episodio
        if done and self.position != 0:
            price  = float(self.df["close"].iloc[-1])
            reward += self._close(price)

        return self._get_obs(), float(reward), done, False, {
            "position":   self.position,
            "capital":    self.capital,
            "n_trades":   self.n_trades,
        }

    # ── Internos ──────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.idx         = self.window
        self.position    = 0        # -1 short | 0 flat | 1 long
        self.entry_price = 0.0
        self.capital     = self.initial_capital
        self.step_count  = 0
        self.n_trades    = 0

    def _close(self, price: float) -> float:
        """Cierra la posicion actual. Devuelve la recompensa (PnL normalizado)."""
        pnl_pct = (price - self.entry_price) / self.entry_price * self.position
        fee     = config.TRADE_FEE * 2            # entrada + salida

        # Actualizar capital
        self.capital *= (1 + pnl_pct - fee)

        # Recompensa = PnL neto (escala razonable para la red)
        reward = float(pnl_pct - fee)

        self.position    = 0
        self.entry_price = 0.0
        self.n_trades   += 1
        return reward

    def _get_obs(self) -> np.ndarray:
        """
        Estado = ultimas WINDOW velas normalizadas.
        Usamos retornos relativos (no precios absolutos) para que el agente
        sea agnóstico al nivel de precios — aprende patrones, no niveles.
        """
        w   = self.df.iloc[self.idx - self.window: self.idx]
        c   = w["close"].values
        eps = 1e-9

        # Retorno porcentual de cada vela
        ret    = np.diff(c, prepend=c[0]) / (c + eps)
        # Alto relativo al cierre
        h_rel  = (w["high"].values  - c) / (c + eps)
        # Bajo relativo al cierre
        l_rel  = (w["low"].values   - c) / (c + eps)
        # Apertura relativa al cierre
        o_rel  = (w["open"].values  - c) / (c + eps)
        # Volumen normalizado por media movil
        vol    = w["volume"].values
        vol_ma = np.convolve(vol, np.ones(20)/20, mode="same") + eps
        v_rel  = vol / vol_ma

        features = np.column_stack([ret, h_rel, l_rel, o_rel, v_rel]).flatten()

        # Estado de posicion: encoding y PnL latente
        price   = c[-1]
        pos_enc = float(self.position)           # -1, 0 o +1
        pnl_lat = 0.0
        if self.position != 0 and self.entry_price > 0:
            pnl_lat = (price - self.entry_price) / self.entry_price * self.position

        obs = np.concatenate([features, [pos_enc, pnl_lat]]).astype(np.float32)
        # Clip para evitar NaN/Inf en los primeros pasos
        return np.clip(obs, -10.0, 10.0)
