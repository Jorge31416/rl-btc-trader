import math
import requests
import logging
import numpy as np
import config

log = logging.getLogger(__name__)


def _send(text: str):
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID,
                  "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception as e:
        log.warning(f"Telegram error: {e}")


# ── Métricas ──────────────────────────────────────────────────────────────────

def _compute_metrics(pnl_usdt: list) -> dict:
    """
    Calcula todas las métricas a partir de lista de PnL en USDT por trade.
    """
    arr   = np.array(pnl_usdt, dtype=float)
    n     = len(arr)
    wins  = arr[arr > 0]
    loses = arr[arr < 0]

    wr    = len(wins) / n if n > 0 else 0.0
    total = float(arr.sum())

    # Profit Factor
    gross_win  = float(wins.sum())  if len(wins)  > 0 else 0.0
    gross_loss = float(-loses.sum()) if len(loses) > 0 else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy por trade
    expect = total / n if n > 0 else 0.0

    # Medias de ganancias y pérdidas
    avg_win  = float(wins.mean())  if len(wins)  > 0 else 0.0
    avg_loss = float(loses.mean()) if len(loses) > 0 else 0.0

    # Sharpe (sobre retornos en USDT, anualizado aprox)
    std = float(arr.std()) if n > 1 else 1e-9
    sharpe = (arr.mean() / std) * math.sqrt(252) if std > 0 else 0.0

    # Calmar = retorno_total / max_drawdown
    equity  = config.INITIAL_CAP + np.cumsum(arr)
    peak    = np.maximum.accumulate(equity)
    dd      = (equity - peak) / (peak + 1e-9)
    max_dd  = float(abs(dd.min())) if n > 0 else 1e-9
    calmar  = (total / config.INITIAL_CAP) / max_dd if max_dd > 1e-9 else 0.0

    # Z-score: test si WR es significativamente distinto de 50%
    z = (wr - 0.5) * 2 * math.sqrt(n) if n > 0 else 0.0
    if abs(z) >= 2.58:
        z_label = "99% significancia"
    elif abs(z) >= 1.96:
        z_label = "95% significancia"
    elif abs(z) >= 1.65:
        z_label = "90% significancia"
    else:
        z_label = "sin significancia"

    return {
        "n": n, "wr": wr, "pf": pf,
        "total": total, "expect": expect,
        "sharpe": sharpe, "calmar": calmar,
        "z": z, "z_label": z_label,
        "avg_win": avg_win, "avg_loss": avg_loss,
    }


def notify_live_summary(pnl_usdt: list, step: int, epsilon: float):
    """Resumen con métricas completas — enviado cada 100 steps."""
    if len(pnl_usdt) < 2:
        _send(f"<b>Resumen (step {step:,})</b>\n"
              f"Trades : {len(pnl_usdt)} — acumulando datos...\n"
              f"Epsilon: {epsilon:.4f}")
        return

    m = _compute_metrics(pnl_usdt)

    _send(
        f"<b>Resumen — step {step:,}</b>\n"
        f"\n"
        f"<b>Acumulado historico:</b>\n"
        f"  Trades : {m['n']} | WR: {m['wr']*100:.1f}% | PF: {m['pf']:.2f}\n"
        f"  PnL    : {m['total']:+.2f} USDT\n"
        f"  Expect : {m['expect']:+.2f} USDT/trade\n"
        f"  Sharpe : {m['sharpe']:.3f} | Calmar: {m['calmar']:.2f}\n"
        f"  Z-score: {m['z']:.2f} ~ ({m['z_label']})\n"
        f"  Media  : {m['avg_win']:+.2f} gan · {m['avg_loss']:+.2f} perd\n"
        f"\n"
        f"Epsilon: {epsilon:.4f}"
    )


# ── Entrenamiento offline ─────────────────────────────────────────────────────

def notify_train_start(candles: int, epochs: int, state_size: int, device: str):
    _send(
        f"<b>Entrenamiento iniciado</b>\n"
        f"Velas     : {candles:,}\n"
        f"Epocas    : {epochs}\n"
        f"Estado    : {state_size} features\n"
        f"Device    : {device}\n"
        f"Acciones  : FLAT / LONG / SHORT"
    )


def notify_train_epoch(ep: int, total: int, capital: float,
                       ret_pct: float, n_trades: int,
                       loss: float, epsilon: float, best_capital: float):
    bar_filled = round(ep / total * 10)
    bar = "█" * bar_filled + "░" * (10 - bar_filled)
    _send(
        f"<b>Epoca {ep}/{total}</b>  [{bar}]\n"
        f"Capital   : {capital:.2f} USDT ({ret_pct:+.1f}%)\n"
        f"Trades    : {n_trades}\n"
        f"Loss      : {loss:.5f}\n"
        f"Epsilon   : {epsilon:.4f}\n"
        f"Mejor cap : {best_capital:.2f} USDT"
    )


def notify_train_end(best_capital: float, final_epsilon: float,
                     total_steps: int, epochs: int):
    ret = (best_capital - config.INITIAL_CAP) / config.INITIAL_CAP * 100
    _send(
        f"<b>Entrenamiento completado</b>\n"
        f"Epocas    : {epochs}\n"
        f"Mejor cap : {best_capital:.2f} USDT ({ret:+.1f}%)\n"
        f"Epsilon   : {final_epsilon:.4f}\n"
        f"Steps     : {total_steps:,}\n\n"
        f"Lanza el bot en vivo con:\n<code>python live.py</code>"
    )


# ── Bot en vivo ───────────────────────────────────────────────────────────────

def notify_start(epsilon: float, steps: int):
    _send(
        f"<b>RL BTC Bot iniciado</b>\n"
        f"Epsilon   : {epsilon:.4f}\n"
        f"Steps prev: {steps:,}"
    )


def notify_position_open(side: str, price: float, epsilon: float, step: int):
    emoji = "L" if side == "LONG" else "S"
    _send(
        f"[{emoji}] <b>Posicion abierta: {side}</b>\n"
        f"Precio  : {price:,.2f} USDT\n"
        f"Epsilon : {epsilon:.4f} | Step: {step:,}"
    )


def notify_trade(side: str, entry: float, exit_price: float,
                 pnl_usdt: float, pnl_pct: float, n_trades: int):
    sign = "+" if pnl_usdt >= 0 else ""
    _send(
        f"<b>Trade #{n_trades} cerrado [{side}]</b>\n"
        f"Entrada: {entry:,.2f} -> {exit_price:,.2f}\n"
        f"PnL    : {sign}{pnl_usdt:.2f} USDT ({pnl_pct*100:+.2f}%)"
    )


def notify_error(msg: str):
    _send(f"<b>Error</b>\n<code>{msg[:400]}</code>")
