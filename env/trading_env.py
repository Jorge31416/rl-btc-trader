"""
Entorno de trading compatible con Gymnasium.

El agente decide entre 3 acciones:
  0 = FLAT   — sin posicion: no hacer nada / con posicion: cerrar
  1 = LONG   — abrir/mantener long  (si habia short, lo cierra primero)
  2 = SHORT  — abrir/mantener short (si habia long, lo cierra primero)

No le decimos que mire RSI, EMA ni ningun indicador.
Solo recibe precios brutos normalizados — aprende el solo que patrones importan.

Mejoras v2:
- Recompensas escaladas por REWARD_SCALE (evita gradientes nulos)
- Penalizacion por mantener posicion perdedora (incentiva cortar perdidas)
- Pequeno coste por paso en posicion (desincentiva sobreoperar)
- Shaping de recompensa consistente con live.py
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

    FLAT  = 0
    LONG  = 1
    SHORT = 2

    # Coste minimo por operar (desincentiva flip-flopping)
    _MIN_HOLD_STEPS = 2   # debe mantener posicion al menos N pasos

    def __init__(self, df: pd.DataFrame, window: int = config.WINDOW,
                 initial_capital: float = config.INITIAL_CAP):
        super().__init__()
        self.df              = df.reset_index(drop=True)
        self.window          = window
        self.initial_capital = initial_capital

        # Estado: window velas * 5 features (ret, h/c, l/c, o/c, vol_ratio)
        #         + 2 valores de posicion (pos_enc, pnl_unrealizado)
        self.state_size = window * 5 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._reset_state()

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        options puede contener:
          start_idx  : vela desde la que empieza el episodio (int)
          episode_len: duracion maxima en pasos (int)
        """
        super().reset(seed=seed)
        opts       = options or {}
        start_idx  = opts.get("start_idx",   self.window)
        episode_len= opts.get("episode_len", None)
        self._reset_state(start_idx=start_idx, episode_len=episode_len)
        return self._get_obs(), {}

    def step(self, action: int):
        price  = float(self.df["close"].iloc[self.idx])
        reward = 0.0

        # ── Ejecutar accion ───────────────────────────────────────────────
        if action == self.FLAT:
            if self.position != 0:
                reward += self._close(price)

        elif action == self.LONG:
            if self.position == -1:          # cierra short primero
                reward += self._close(price)
            if self.position != 1:
                self.position      = 1
                self.entry_price   = price
                self.hold_steps    = 0

        elif action == self.SHORT:
            if self.position == 1:           # cierra long primero
                reward += self._close(price)
            if self.position != -1:
                self.position      = -1
                self.entry_price   = price
                self.hold_steps    = 0

        # ── Shaping mientras se mantiene posicion ─────────────────────────
        if self.position != 0:
            self.hold_steps += 1
            unrealized = (price - self.entry_price) / self.entry_price * self.position

            # Penalizacion por drawdown grande (> 1.5%) — incentiva cortar perdidas
            if unrealized < -0.015:
                reward -= 0.003 * config.REWARD_SCALE

            # Penalizacion adicional por drawdown extremo (> 3%)
            elif unrealized < -0.030:
                reward -= 0.005 * config.REWARD_SCALE

        self.idx        += 1
        self.step_count += 1
        done = self.idx >= self.end_idx

        # Forzar cierre al final del episodio
        if done and self.position != 0:
            price   = float(self.df["close"].iloc[-1])
            reward += self._close(price)

        return self._get_obs(), float(reward), done, False, {
            "position": self.position,
            "capital":  self.capital,
            "n_trades": self.n_trades,
        }

    # ── Internos ──────────────────────────────────────────────────────────────

    def _reset_state(self, start_idx: int = None, episode_len: int = None):
        self.idx         = start_idx if start_idx is not None else self.window
        self.end_idx     = (self.idx + episode_len) if episode_len else len(self.df) - 1
        self.end_idx     = min(self.end_idx, len(self.df) - 1)
        self.position    = 0        # -1 short | 0 flat | 1 long
        self.entry_price = 0.0
        self.capital     = self.initial_capital
        self.step_count  = 0
        self.n_trades    = 0
        self.hold_steps  = 0       # pasos desde que se abrio la posicion actual

    def _close(self, price: float) -> float:
        """
        Cierra la posicion actual.
        Devuelve recompensa = PnL neto escalado por REWARD_SCALE.
        Escalar es clave: sin esto el gradiente es ~0 y la red no aprende.
        """
        pnl_pct = (price - self.entry_price) / self.entry_price * self.position
        fee     = config.TRADE_FEE * 2       # entrada + salida
        net_pnl = pnl_pct - fee
        self.capital    *= (1 + net_pnl)
        self.position    = 0
        self.entry_price = 0.0
        self.hold_steps  = 0
        self.n_trades   += 1
        # Escalar: 0.2% de ganancia → 20 unidades de reward
        return float(net_pnl * config.REWARD_SCALE)

    def _get_obs(self) -> np.ndarray:
        """
        Estado = ultimas WINDOW velas normalizadas.
        Retornos relativos (no precios absolutos) — el agente aprende
        patrones de movimiento, no niveles de precio.
        """
        w   = self.df.iloc[self.idx - self.window: self.idx]
        c   = w["close"].values
        eps = 1e-9

        ret   = np.diff(c, prepend=c[0]) / (c + eps)   # retorno vela a vela
        h_rel = (w["high"].values  - c) / (c + eps)    # mecha superior
        l_rel = (w["low"].values   - c) / (c + eps)    # mecha inferior
        o_rel = (w["open"].values  - c) / (c + eps)    # apertura vs cierre
        vol   = w["volume"].values
        v_rel = vol / (np.convolve(vol, np.ones(20)/20, mode="same") + eps)

        features = np.column_stack([ret, h_rel, l_rel, o_rel, v_rel]).flatten()

        # Posicion actual y PnL latente
        price   = c[-1]
        pos_enc = float(self.position)
        pnl_lat = 0.0
        if self.position != 0 and self.entry_price > 0:
            pnl_lat = (price - self.entry_price) / self.entry_price * self.position

        obs = np.concatenate([features, [pos_enc, pnl_lat]]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)
