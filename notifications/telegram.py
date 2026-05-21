import requests
import logging
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
    emoji = "+" if ret_pct >= 0 else "-"
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

def notify_position_open(side: str, price: float, epsilon: float, step: int):
    emoji = "📈" if side == "LONG" else "📉"
    _send(
        f"{emoji} <b>Posicion abierta: {side}</b>\n"
        f"Precio  : {price:,.2f} USDT\n"
        f"Epsilon : {epsilon:.4f}\n"
        f"Step    : {step:,}"
    )


def notify_start(epsilon: float, steps: int):
    _send(
        f"<b>RL BTC Bot iniciado</b>\n"
        f"Epsilon   : {epsilon:.3f}\n"
        f"Steps prev: {steps:,}"
    )


def notify_trade(side: str, entry: float, exit_price: float,
                 pnl_pct: float, n_trades: int, epsilon: float):
    emoji = "+" if pnl_pct > 0 else "-"
    _send(
        f"<b>Trade #{n_trades} cerrado</b>\n"
        f"Lado   : {side.upper()}\n"
        f"Entrada: {entry:,.2f}\n"
        f"Salida : {exit_price:,.2f}\n"
        f"PnL    : {pnl_pct*100:+.2f}% {emoji}\n"
        f"Epsilon: {epsilon:.4f}"
    )


def notify_summary(step: int, n_trades: int, win_rate: float,
                   total_return: float, epsilon: float):
    _send(
        f"<b>Resumen diario</b>\n"
        f"Steps     : {step:,}\n"
        f"Trades    : {n_trades}\n"
        f"Win rate  : {win_rate*100:.1f}%\n"
        f"Retorno   : {total_return*100:+.2f}%\n"
        f"Epsilon   : {epsilon:.4f}"
    )


def notify_error(msg: str):
    _send(f"<b>Error</b>\n<code>{msg[:400]}</code>")
