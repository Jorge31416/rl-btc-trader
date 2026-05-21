"""
Entrenamiento OFFLINE — pre-entrena el agente sobre datos historicos.

Antes de lanzar el bot en vivo, hay que darle miles de episodios
para que el agente salga del comportamiento completamente aleatorio.
Es equivalente a dejar que el bot de Invaders juegue miles de partidas
en modo acelerado antes de competir en serio.

Uso:
    python train.py
    python train.py --candles 10000 --epochs 50
"""
import sys
import logging
import argparse
import time

import numpy as np
import pandas as pd

import config
from data.demo_client import BinanceDemoClient
from env.trading_env import TradingEnv
from agent.dqn import DQNAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def download_history(candles: int) -> pd.DataFrame:
    """
    Descarga datos historicos paginando en chunks de 1500
    (limite maximo de Binance por request).
    """
    MAX_PER_REQUEST = 1500
    log.info(f"Descargando {candles} velas de {config.SYMBOL} {config.TIMEFRAME}...")
    client = BinanceDemoClient(config.API_KEY, config.API_SECRET)

    all_rows = []
    end_time = None   # None = desde ahora hacia atras

    while len(all_rows) < candles:
        batch_size = min(MAX_PER_REQUEST, candles - len(all_rows))
        raw = client.fetch_ohlcv(
            config.SYMBOL, config.TIMEFRAME,
            limit=batch_size, end_time=end_time,
        )
        if not raw:
            break
        # raw viene en orden ascendente; el primer elemento es el mas antiguo
        all_rows = raw + all_rows          # prepend → orden cronologico
        end_time = raw[0][0] - 1           # siguiente batch termina antes del primero actual
        log.info(f"  Descargadas {len(all_rows)}/{candles} velas...")
        if len(raw) < batch_size:          # Binance devolvio menos de lo pedido → no hay mas
            break

    df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated()].sort_index()
    log.info(f"Datos: {len(df)} velas | {df.index[0]} -> {df.index[-1]}")
    return df.astype(float)


def run_episode(env: TradingEnv, agent: DQNAgent,
                training: bool = True) -> dict:
    """Ejecuta un episodio completo y devuelve metricas."""
    obs, _    = env.reset()
    total_rew = 0.0
    losses    = []

    while True:
        action              = agent.act(obs, training=training)
        next_obs, rew, done, _, info = env.step(action)

        if training:
            agent.remember(obs, action, rew, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        total_rew += rew
        obs        = next_obs

        if done:
            break

    return {
        "reward":   total_rew,
        "capital":  info["capital"],
        "n_trades": info["n_trades"],
        "loss":     float(np.mean(losses)) if losses else 0.0,
        "epsilon":  agent.epsilon,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candles", type=int, default=config.PRETRAIN_CANDLES)
    parser.add_argument("--epochs",  type=int, default=config.PRETRAIN_EPOCHS)
    parser.add_argument("--resume",  action="store_true",
                        help="Cargar checkpoint existente y continuar")
    args = parser.parse_args()

    df  = download_history(args.candles)
    env = TradingEnv(df)
    agent = DQNAgent(state_size=env.state_size)

    if args.resume:
        try:
            agent.load()
        except FileNotFoundError:
            log.info("No hay checkpoint — empezando desde cero")

    log.info("=" * 55)
    log.info("  ENTRENAMIENTO OFFLINE")
    log.info(f"  Velas: {len(df)}  Epocas: {args.epochs}")
    log.info(f"  Estado: {env.state_size} features")
    log.info(f"  Device: {agent.device}")
    log.info("=" * 55)

    best_capital = 0.0

    for ep in range(1, args.epochs + 1):
        t0      = time.time()
        metrics = run_episode(env, agent, training=True)

        capital     = metrics["capital"]
        total_rew   = metrics["reward"]
        n_trades    = metrics["n_trades"]
        loss        = metrics["loss"]
        epsilon     = metrics["epsilon"]
        ret_pct     = (capital - config.INITIAL_CAP) / config.INITIAL_CAP * 100
        elapsed     = time.time() - t0

        log.info(
            f"Ep {ep:3d}/{args.epochs} | "
            f"Capital: {capital:.2f} ({ret_pct:+.1f}%) | "
            f"Trades: {n_trades:3d} | "
            f"Reward: {total_rew:+.3f} | "
            f"Loss: {loss:.4f} | "
            f"Eps: {epsilon:.3f} | "
            f"{elapsed:.1f}s"
        )

        # Guardar si es el mejor resultado
        if capital > best_capital:
            best_capital = capital
            agent.save("checkpoints/dqn_best.pth")

        # Checkpoint periodico
        if ep % 10 == 0:
            agent.save("checkpoints/dqn_latest.pth")

    agent.save("checkpoints/dqn_final.pth")
    log.info("=" * 55)
    log.info(f"  Entrenamiento completado.")
    log.info(f"  Mejor capital: {best_capital:.2f} USDT")
    log.info(f"  Epsilon final: {agent.epsilon:.4f}")
    log.info("=" * 55)
    log.info("Ahora ejecuta: python live.py")


if __name__ == "__main__":
    main()
