"""
Entrenamiento OFFLINE — pre-entrena el agente sobre datos historicos.

Antes de lanzar el bot en vivo, hay que darle miles de episodios
para que el agente salga del comportamiento completamente aleatorio.
Es equivalente a dejar que el bot de Invaders juegue miles de partidas
en modo acelerado antes de competir en serio.

Mejoras v2:
- 20k velas por defecto (~10 semanas de datos 5m)
- 100 epocas por defecto
- Recompensas escaladas (corrige loss=0 del modelo anterior)
- Dueling DQN para mejor separacion valor/accion
- Al final resetea epsilon a EPSILON_LIVE_START (exploracion en vivo)

Uso:
    python train.py                          # entrena desde cero (recomendado)
    python train.py --candles 30000 --epochs 150
    python train.py --resume                 # continua desde checkpoint existente
"""
import sys
import random
import logging
import argparse
import time
from pathlib import Path

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
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def download_history(candles: int) -> pd.DataFrame:
    """
    Descarga datos historicos paginando en chunks de 1500
    (limite maximo de Binance por request).
    """
    MAX_PER_REQUEST = 1500
    log.info(f"Descargando {candles:,} velas de {config.SYMBOL} {config.TIMEFRAME}...")
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
        log.info(f"  {len(all_rows):,}/{candles:,} velas descargadas...")
        if len(raw) < batch_size:          # Binance devolvio menos de lo pedido
            break

    df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated()].sort_index()
    log.info(f"Datos: {len(df):,} velas | {df.index[0]} → {df.index[-1]}")
    return df.astype(float)


SUB_EPISODE_LEN = 500   # pasos por sub-episodio — con 500 pasos aleatorios
                        # el agente nunca puede quebrar por fees
SUBS_PER_EPOCH  = 100  # sub-episodios por epoca — 100x500 = 50k pasos/epoca


def run_epoch(env: TradingEnv, agent: DQNAgent,
              training: bool = True, train_every: int = 4) -> dict:
    """
    Ejecuta una epoca = SUBS_PER_EPOCH sub-episodios cortos sobre ventanas
    aleatorias de los datos historicos.

    Ventajas vs un episodio unico de 200k pasos:
    - Nunca hay bancarrota por fees en exploracion aleatoria
    - El agente ve condiciones de mercado diversas en cada epoch
    - Equivalente a "jugar muchas partidas cortas" como Atari
    - ~4x mas rapido que recorrer los 200k pasos completos
    """
    max_start   = len(env.df) - SUB_EPISODE_LEN - env.window - 2
    capitals    = []
    total_trades= 0
    losses      = []

    for _ in range(SUBS_PER_EPOCH):
        start = random.randint(env.window, max_start)
        obs, _ = env.reset(options={"start_idx": start,
                                    "episode_len": SUB_EPISODE_LEN})
        step_n = 0

        while True:
            action               = agent.act(obs, training=training)
            next_obs, rew, done, _, info = env.step(action)

            if training:
                agent.remember(obs, action, rew, next_obs, done)
                if step_n % train_every == 0:
                    loss = agent.train_step(decay_epsilon=False)
                    if loss is not None:
                        losses.append(loss)

            obs     = next_obs
            step_n += 1
            if done:
                break

        capitals.append(info["capital"])
        total_trades += info["n_trades"]

    return {
        "capital":  float(np.mean(capitals)),
        "n_trades": total_trades // SUBS_PER_EPOCH,  # media por sub-episodio
        "loss":     float(np.mean(losses)) if losses else 0.0,
        "epsilon":  agent.epsilon,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candles", type=int, default=config.PRETRAIN_CANDLES)
    parser.add_argument("--epochs",  type=int, default=config.PRETRAIN_EPOCHS)
    parser.add_argument("--resume",  action="store_true",
                        help="Continuar desde checkpoint existente")
    args = parser.parse_args()

    # Limpiar checkpoints antiguos si se empieza de cero
    # (arquitectura Dueling DQN es incompatible con checkpoints v1)
    if not args.resume:
        for ckpt in ["checkpoints/dqn_best.pth",
                     "checkpoints/dqn_latest.pth",
                     "checkpoints/dqn_final.pth"]:
            if Path(ckpt).exists():
                Path(ckpt).unlink()
                log.info(f"Checkpoint v1 eliminado: {ckpt}")

    df    = download_history(args.candles)
    env   = TradingEnv(df)
    agent = DQNAgent(state_size=env.state_size)

    if args.resume:
        for ckpt in ["checkpoints/dqn_best.pth", "checkpoints/dqn_latest.pth"]:
            if Path(ckpt).exists():
                agent.load(ckpt)
                log.info(f"Continuando desde {ckpt}")
                break
        else:
            log.info("No hay checkpoint — empezando desde cero")

    log.info("=" * 60)
    log.info("  ENTRENAMIENTO OFFLINE v2 (Dueling DQN)")
    log.info(f"  Velas   : {len(df):,}  ({df.index[0].date()} → {df.index[-1].date()})")
    log.info(f"  Epocas  : {args.epochs}")
    log.info(f"  Estado  : {env.state_size} features")
    log.info(f"  Device  : {agent.device}")
    log.info(f"  Reward  : PnL × {config.REWARD_SCALE:.0f}")
    log.info(f"  Epsilon : {agent.epsilon:.3f} → {config.EPSILON_MIN}")
    log.info("=" * 60)

    tg.notify_train_start(len(df), args.epochs,
                          env.state_size, str(agent.device))

    best_capital  = 0.0
    best_state    = None      # pesos del mejor modelo
    SUMMARY_EVERY = 5         # Telegram cada N epocas
    t_start       = time.time()

    # Epsilon decae linealmente de 1.0 a EPSILON_MIN a lo largo del 70% de las epocas.
    # El 30% final entrena con epsilon minimo (explotacion pura).
    # Esto evita que epsilon caiga a 0.05 en los primeros 6000 pasos del episodio 1.
    DECAY_UNTIL = int(args.epochs * 0.7)

    for ep in range(1, args.epochs + 1):
        progress      = min(1.0, (ep - 1) / max(1, DECAY_UNTIL))
        agent.epsilon = 1.0 - progress * (1.0 - config.EPSILON_MIN)

        t0      = time.time()
        metrics = run_epoch(env, agent, training=True)

        capital   = metrics["capital"]
        n_trades  = metrics["n_trades"]
        loss      = metrics["loss"]
        epsilon   = metrics["epsilon"]
        ret_pct   = (capital - config.INITIAL_CAP) / config.INITIAL_CAP * 100
        elapsed   = time.time() - t0
        eta_min   = (args.epochs - ep) * elapsed / 60

        log.info(
            f"Ep {ep:3d}/{args.epochs} | "
            f"Cap: {capital:.0f} ({ret_pct:+.1f}%) | "
            f"Trades: {n_trades:3d} | "
            f"Loss: {loss:.4f} | "
            f"Eps: {epsilon:.4f} | "
            f"{elapsed:.1f}s | ETA: {eta_min:.0f}min"
        )

        # Guardar pesos del mejor episodio
        if capital > best_capital:
            best_capital = capital
            # Guardar copia de los pesos (no del epsilon — se resetea al final)
            best_state = {
                "q_net":      {k: v.cpu().clone() for k, v in agent.q_net.state_dict().items()},
                "target_net": {k: v.cpu().clone() for k, v in agent.target_net.state_dict().items()},
                "optimizer":  agent.optimizer.state_dict(),
                "steps":      agent.steps,
            }
            log.info(f"  ★ Nuevo mejor capital: {best_capital:.2f} USDT ({ret_pct:+.1f}%)")

        # Checkpoint + Telegram cada SUMMARY_EVERY epocas
        if ep % SUMMARY_EVERY == 0 or ep == args.epochs:
            agent.save("checkpoints/dqn_latest.pth")
            tg.notify_train_epoch(ep, args.epochs, capital, ret_pct,
                                  n_trades, loss, epsilon, best_capital)

    # ── Preparar checkpoint para live ────────────────────────────────────────
    # Cargar los MEJORES pesos, resetear epsilon a EPSILON_LIVE_START
    # para que el bot vivo empiece con exploracion, no solo explotacion.
    log.info("")
    log.info(f"Preparando checkpoint para live (eps → {config.EPSILON_LIVE_START})...")

    if best_state is not None:
        agent.q_net.load_state_dict(
            {k: v.to(agent.device) for k, v in best_state["q_net"].items()}
        )
        agent.target_net.load_state_dict(
            {k: v.to(agent.device) for k, v in best_state["target_net"].items()}
        )
        agent.steps = best_state["steps"]

    agent.epsilon = config.EPSILON_LIVE_START
    agent.save("checkpoints/dqn_best.pth")

    elapsed_total = (time.time() - t_start) / 60
    tg.notify_train_end(best_capital, config.EPSILON_LIVE_START,
                        agent.steps, args.epochs)

    log.info("=" * 60)
    log.info(f"  Entrenamiento completado en {elapsed_total:.0f} min")
    log.info(f"  Mejor capital: {best_capital:.2f} USDT")
    log.info(f"  Epsilon live : {config.EPSILON_LIVE_START}")
    log.info(f"  Steps totales: {agent.steps:,}")
    log.info("=" * 60)
    log.info("Ahora ejecuta: python live.py")


if __name__ == "__main__":
    main()
