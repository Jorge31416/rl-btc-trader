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


def notify_start(epsilon: float, steps: int):
    _send(f"<b>RL BTC Bot iniciado</b>\n"
          f"Epsilon: {epsilon:.3f} | Steps previos: {steps}")


def notify_trade(side: str, entry: float, exit_price: float,
                 pnl_pct: float, reason: str, n_trades: int, epsilon: float):
    emoji = "+" if pnl_pct > 0 else "-"
    _send(f"<b>Trade #{n_trades} cerrado</b>\n"
          f"Lado   : {side.upper()}\n"
          f"Entrada: {entry:.2f}\n"
          f"Salida : {exit_price:.2f}\n"
          f"PnL    : {pnl_pct*100:+.2f}% {emoji}\n"
          f"Motivo : {reason}\n"
          f"Epsilon: {epsilon:.3f}")


def notify_summary(episode: int, n_trades: int, win_rate: float,
                   total_return: float, epsilon: float):
    _send(f"<b>Resumen episodio {episode}</b>\n"
          f"Trades    : {n_trades}\n"
          f"Win rate  : {win_rate*100:.1f}%\n"
          f"Retorno   : {total_return*100:+.2f}%\n"
          f"Epsilon   : {epsilon:.3f}")


def notify_error(msg: str):
    _send(f"<b>Error</b>\n<code>{msg[:400]}</code>")
