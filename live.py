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
import pickle
import logging
import argparse
from pathlib import Path
from collections import deque
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

    BUFFER_PATH = Path("checkpoints/replay_buffer.pkl")

    if not args.fresh:
        for ckpt in ["checkpoints/dqn_best.pth", "checkpoints/dqn_latest.pth"]:
            if Path(ckpt).exists():
                agent.load(ckpt)
                break
        else:
            log.info("Sin checkpoint previo — empezando con epsilon=1.0 (puro exploracion)")
            log.info("Consejo: ejecuta primero  python train.py  para pre-entrenar")

        # Cargar replay buffer persistido (si existe)
        if BUFFER_PATH.exists():
            try:
                with open(BUFFER_PATH, "rb") as f:
                    saved = pickle.load(f)
                agent.buffer.buf = deque(saved, maxlen=config.BUFFER_SIZE)
                log.info(f"Buffer cargado: {len(agent.buffer):,} experiencias previas")
            except Exception as e:
                log.warning(f"No se pudo cargar buffer: {e}")

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
    entry_balance  = 0.0      # balance USDT al abrir la posicion
    n_trades       = 0
    pnl_usdt_hist  = []       # PnL en USDT por trade (para métricas)
    prev_obs       = None
    prev_action    = None
    prev_position  = 0        # posicion en el paso anterior (para calcular recompensa)
    pending_reward = 0.0      # recompensa del cierre pendiente de enviar al buffer
    step           = 0

    while True:
        try:
            df = fetch_df(client)
            if df is None or len(df) < config.WINDOW + 1:
                time.sleep(30)
                continue

            price = float(df["close"].iloc[-1])
            obs   = get_obs(df, env)

            # Actualizar estado del entorno con posicion real
            env.position    = live_position
            env.entry_price = entry_price

            # Recompensa del paso anterior → buffer
            # REWARD_SCALE = 100 escala las recompensas para que los gradientes no sean nulos.
            # El cierre se calcula en el mismo tick donde entry_price todavia es valido,
            # y se guarda en pending_reward para aplicar en el buffer del tick siguiente.
            if prev_obs is not None and prev_action is not None:
                if pending_reward != 0.0:
                    # Recompensa de cierre calculada el tick anterior
                    raw_reward    = pending_reward
                    pending_reward = 0.0
                elif live_position != 0:
                    # En posicion: penalizar drawdown grande (consistente con env)
                    unreal = (price - entry_price) / entry_price * live_position
                    raw_reward = -0.003 * config.REWARD_SCALE if unreal < -0.015 else 0.0
                else:
                    raw_reward = 0.0
                agent.remember(prev_obs, prev_action, raw_reward, obs, False)

            # Decidir accion
            action     = agent.act(obs, training=True)
            action_str = agent.ACTION_NAMES[action]

            # Calcular PnL latente para el log
            unreal = 0.0
            if live_position != 0 and entry_price > 0:
                unreal = (price - entry_price) / entry_price * live_position

            pos_char = "L" if live_position == 1 else ("S" if live_position == -1 else "-")
            log.info(
                f"Price={price:.2f} | Pos={pos_char}"
                f"{f' PnL={unreal*100:+.2f}%' if live_position != 0 else ''} | "
                f"Action={action_str} | Eps={agent.epsilon:.4f} | "
                f"Buf={len(agent.buffer)} | Trades={n_trades}"
            )

            # Ejecutar accion
            prev_position = live_position
            live_position = execute_action(client, action, live_position, price)

            # ── Apertura de posicion ──────────────────────────────────────
            if prev_position == 0 and live_position != 0:
                entry_price   = price
                try:
                    entry_balance = client.fetch_balance()["USDT"]["total"]
                except Exception:
                    entry_balance = config.INITIAL_CAP
                side_str = "LONG" if live_position == 1 else "SHORT"
                log.info(f"[OPEN] {side_str} @ {price:.2f}")
                tg.notify_position_open(
                    side    = side_str,
                    price   = price,
                    epsilon = agent.epsilon,
                    step    = step,
                )

            # ── Cierre de posicion ────────────────────────────────────────
            elif prev_position != 0 and live_position == 0:
                pnl_pct  = (price - entry_price) / entry_price * prev_position
                fee      = config.TRADE_FEE * 2
                pnl_net  = pnl_pct - fee
                pnl_usdt = pnl_net * entry_balance
                n_trades += 1
                pnl_usdt_hist.append(pnl_usdt)

                # Guardar recompensa de cierre para el buffer del proximo tick
                # (entry_price aun es valido aqui antes de resetearlo)
                pending_reward = pnl_net * config.REWARD_SCALE

                log.info(f"[CLOSE] Trade #{n_trades} | "
                         f"PnL={pnl_net*100:+.2f}% ({pnl_usdt:+.2f} USDT)")
                tg.notify_trade(
                    side       = "LONG" if prev_position == 1 else "SHORT",
                    entry      = entry_price,
                    exit_price = price,
                    pnl_usdt   = pnl_usdt,
                    pnl_pct    = pnl_net,
                    n_trades   = n_trades,
                )
                entry_price   = 0.0
                entry_balance = 0.0

            prev_obs    = obs
            prev_action = action
            step       += 1

            # ── Entrenamiento periodico ───────────────────────────────────
            if step % config.TRAIN_EVERY_STEPS == 0 and len(agent.buffer) >= config.BATCH_SIZE:
                losses = []
                for _ in range(10):
                    l = agent.train_step()
                    if l is not None:
                        losses.append(l)
                if losses:
                    log.info(f"[TRAIN] loss={np.mean(losses):.5f} eps={agent.epsilon:.4f}")

            # ── Checkpoint periodico ──────────────────────────────────────
            if step % config.SAVE_EVERY_STEPS == 0:
                agent.save("checkpoints/dqn_latest.pth")
                # Guardar buffer en disco — sobrevive reinicios
                try:
                    with open(BUFFER_PATH, "wb") as f:
                        pickle.dump(list(agent.buffer.buf), f)
                    log.info(f"Checkpoint guardado (step {step}) | Buffer: {len(agent.buffer):,} exp")
                except Exception as e:
                    log.warning(f"No se pudo guardar buffer: {e}")
                    log.info(f"Checkpoint guardado (step {step})")

            # ── Resumen cada 100 steps ───────────────────────────────────
            if step % 100 == 0:
                tg.notify_live_summary(pnl_usdt_hist, step, agent.epsilon)

            time.sleep(config.CHECK_INTERVAL_S)

        except KeyboardInterrupt:
            log.info("Detenido por el usuario. Guardando checkpoint y buffer...")
            agent.save("checkpoints/dqn_latest.pth")
            try:
                with open(BUFFER_PATH, "wb") as f:
                    pickle.dump(list(agent.buffer.buf), f)
                log.info(f"Buffer guardado: {len(agent.buffer):,} experiencias")
            except Exception as e:
                log.warning(f"No se pudo guardar buffer: {e}")
            break
        except Exception as e:
            log.error(f"Error en tick: {e}", exc_info=True)
            tg.notify_error(str(e))
            time.sleep(30)


if __name__ == "__main__":
    main()
