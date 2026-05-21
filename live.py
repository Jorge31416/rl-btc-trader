"""
Bot en vivo — paper trading con aprendizaje continuo.

Cada tick el agente:
  1. Descarga las ultimas LOOKBACK velas
  2. Construye el estado (ultimas WINDOW velas normalizadas)
  3. Decide la accion (HOLD/LONG/SHORT/CLOSE)
  4. Ejecuta en Binance Demo (si posicion cambia)
  5. Guarda la experiencia en el replay buffer
  6. Entrena cada TRAIN_EVERY_STEPS pasos

El agente sigue aprendiendo indefinidamente — igual que el bot de Invaders
que sigue mejorando cuantas mas partidas juega.

Uso:
    python live.py           # empieza desde checkpoint si existe
    python live.py --fresh   # ignora checkpoint, empieza desde cero
"""
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from data.demo_client import BinanceDemoClient
from env.trading_env import TradingEnv
from agent.dqn import DQNAgent
import notifications.telegram as tg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_df(client: BinanceDemoClient) -> pd.DataFrame | None:
    try:
        raw = client.fetch_ohlcv(config.SYMBOL, config.TIMEFRAME,
                                 limit=config.LOOKBACK)
        df  = pd.DataFrame(raw,
                           columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df.astype(float)
    except Exception as e:
        log.error(f"Error descargando datos: {e}")
        return None


def get_obs(df: pd.DataFrame, env: TradingEnv) -> np.ndarray:
    """Extrae el estado de las ultimas WINDOW velas del df en vivo."""
    env.df  = df.reset_index(drop=True)
    env.idx = len(df) - 1          # apunta a la ultima vela
    return env._get_obs()


# ── Ejecucion de ordenes ──────────────────────────────────────────────────────

def execute_action(client: BinanceDemoClient, action: int,
                   current_position: int, price: float) -> int:
    """
    Ejecuta la accion en Binance Demo.
    Devuelve la nueva posicion resultante.
    """
    try:
        sym = config.SYMBOL

        if action == TradingEnv.FLAT:
            if current_position != 0:
                _close_all(client)
                log.info(f"[LIVE] FLAT (cierre) @ {price:.2f}")
            return 0

        elif action == TradingEnv.LONG and current_position != 1:
            if current_position == -1:
                _close_all(client)
            balance = client.fetch_balance()["USDT"]["free"]
            qty = round(balance * 0.95 / price, 3)
            if qty > 0:
                client.create_market_order(sym, "buy", qty)
                log.info(f"[LIVE] LONG {qty} BTC @ {price:.2f}")
            return 1

        elif action == TradingEnv.SHORT and current_position != -1:
            if current_position == 1:
                _close_all(client)
            balance = client.fetch_balance()["USDT"]["free"]
            qty = round(balance * 0.95 / price, 3)
            if qty > 0:
                client.create_market_order(sym, "sell", qty)
                log.info(f"[LIVE] SHORT {qty} BTC @ {price:.2f}")
            return -1

    except Exception as e:
        log.error(f"Error ejecutando orden: {e}")

    return current_position


def _close_all(client: BinanceDemoClient):
    positions = client.fetch_positions([config.SYMBOL])
    for pos in positions:
        side = "sell" if pos["side"] == "long" else "buy"
        client.create_market_order(config.SYMBOL, side, pos["contracts"])


# ── Loop principal ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                        help="Ignorar checkpoint y empezar desde cero")
    args = parser.parse_args()

    client = BinanceDemoClient(config.API_KEY, config.API_SECRET)

    # Crear entorno temporal para calcular state_size
    df0 = fetch_df(client)
    if df0 is None:
        log.error("No se pudo conectar a Binance. Verifica las API keys.")
        sys.exit(1)

    env   = TradingEnv(df0)
    agent = DQNAgent(state_size=env.state_size)

    if not args.fresh:
        for ckpt in ["checkpoints/dqn_best.pth", "checkpoints/dqn_latest.pth"]:
            if Path(ckpt).exists():
                agent.load(ckpt)
                break
        else:
            log.info("Sin checkpoint previo — empezando con epsilon=1.0 (puro exploracion)")
            log.info("Consejo: ejecuta primero  python train.py  para pre-entrenar")

    log.info("=" * 55)
    log.info("  RL BTC BOT — aprendizaje continuo en vivo")
    log.info(f"  Symbol   : {config.SYMBOL} {config.TIMEFRAME}")
    log.info(f"  Estado   : {env.state_size} features")
    log.info(f"  Device   : {agent.device}")
    log.info(f"  Epsilon  : {agent.epsilon:.3f}")
    log.info("=" * 55)

    tg.notify_start(agent.epsilon, agent.steps)

    # Estado del bot
    live_position  = 0        # posicion real en Binance (-1, 0, 1)
    entry_price    = 0.0
    n_trades       = 0
    prev_obs       = None
    prev_action    = None
    step           = 0
    last_summary_h = -1

    while True:
        try:
            df = fetch_df(client)
            if df is None or len(df) < config.WINDOW + 1:
                time.sleep(30)
                continue

            price   = float(df["close"].iloc[-1])
            obs     = get_obs(df, env)

            # Actualizar estado del entorno con posicion real
            env.position    = live_position
            env.entry_price = entry_price

            # Calcular recompensa del paso anterior
            if prev_obs is not None and prev_action is not None:
                # Reward basado en cambio de precio y posicion
                if live_position != 0:
                    raw_reward = (price - entry_price) / entry_price * live_position
                else:
                    raw_reward = 0.0
                # Guardar en buffer
                agent.remember(prev_obs, prev_action, raw_reward, obs,
                               False)  # done=False en live

            # Decidir accion
            action     = agent.act(obs, training=True)
            action_str = agent.ACTION_NAMES[action]

            log.info(
                f"Price={price:.2f} | Pos={'L' if live_position==1 else ('S' if live_position==-1 else '-')} | "
                f"Action={action_str} | Eps={agent.epsilon:.3f} | "
                f"Buf={len(agent.buffer)}"
            )

            # Ejecutar si cambia algo
            prev_position = live_position
            live_position = execute_action(client, action, live_position, price)

            # Registrar apertura/cierre
            if prev_position == 0 and live_position != 0:
                entry_price = price
            elif prev_position != 0 and live_position == 0:
                pnl = (price - entry_price) / entry_price * prev_position
                n_trades += 1
                log.info(f"Trade #{n_trades} cerrado | PnL={pnl*100:+.2f}%")
                tg.notify_trade(
                    side       = "long" if prev_position == 1 else "short",
                    entry      = entry_price,
                    exit_price = price,
                    pnl_pct    = pnl,
                    reason     = action_str,
                    n_trades   = n_trades,
                    epsilon    = agent.epsilon,
                )
                entry_price = 0.0

            prev_obs    = obs
            prev_action = action
            step       += 1

            # Entrenar periodicamente
            if step % config.TRAIN_EVERY_STEPS == 0 and len(agent.buffer) >= config.BATCH_SIZE:
                losses = [agent.train_step() for _ in range(10) if agent.train_step() is not None]
                if losses:
                    log.info(f"[TRAIN] loss_mean={np.mean(losses):.4f} eps={agent.epsilon:.4f}")

            # Guardar checkpoint periodicamente
            if step % config.SAVE_EVERY_STEPS == 0:
                agent.save("checkpoints/dqn_latest.pth")

            # Resumen diario a las 08:00 UTC
            now_h = datetime.now(timezone.utc).hour
            if now_h == 8 and last_summary_h != 8:
                last_summary_h = 8
                try:
                    balance = client.fetch_balance()["USDT"]["total"]
                    ret = (balance - config.INITIAL_CAP) / config.INITIAL_CAP
                    tg.notify_summary(step, n_trades, 0.0, ret, agent.epsilon)
                except Exception:
                    pass
            elif now_h != 8:
                last_summary_h = -1

            time.sleep(config.CHECK_INTERVAL_S)

        except KeyboardInterrupt:
            log.info("Detenido por el usuario. Guardando checkpoint...")
            agent.save("checkpoints/dqn_latest.pth")
            break
        except Exception as e:
            log.error(f"Error en tick: {e}", exc_info=True)
            tg.notify_error(str(e))
            time.sleep(30)


if __name__ == "__main__":
    main()
