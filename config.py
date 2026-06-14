import os
from dotenv import load_dotenv

load_dotenv()

# ── Binance Demo Trading ──────────────────────────────────────────────────────
API_KEY    = os.environ["BINANCE_API_KEY"]
API_SECRET = os.environ["BINANCE_API_SECRET"]

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Market ────────────────────────────────────────────────────────────────────
SYMBOL    = "BTC/USDT:USDT"
TIMEFRAME = "1h"

# ── Entorno de trading ────────────────────────────────────────────────────────
WINDOW        = 50          # velas que ve el agente como estado
INITIAL_CAP   = 3000.0      # capital inicial en USDT
TRADE_FEE     = 0.0004      # 0.04% por operacion (Binance futures)

# ── DQN ───────────────────────────────────────────────────────────────────────
LR               = 3e-4     # lr mas alto converge mas rapido al principio
GAMMA            = 0.99     # descuento de recompensas futuras
EPSILON_START    = 1.0      # exploracion inicial (100% aleatoria)
EPSILON_MIN      = 0.05     # exploracion minima (5%)
EPSILON_DECAY    = 0.9997   # decaimiento mas lento → mejor exploracion offline
BUFFER_SIZE      = 200_000  # buffer mas grande → mas diversidad de experiencias
BATCH_SIZE       = 256      # batches mayores → gradientes mas estables
TARGET_UPDATE    = 1000     # actualizar red objetivo con menos frecuencia

# ── Escala de recompensas ─────────────────────────────────────────────────────
# El PnL de un trade (~0.2%) es muy pequeño para entrenar una red neuronal.
# Multiplicar por 100 da recompensas en rango [-2, +2], mucho mejor para gradientes.
REWARD_SCALE     = 100.0

# ── Entrenamiento offline ─────────────────────────────────────────────────────
PRETRAIN_CANDLES = 40_000   # ~4.5 años de velas 1h — Binance futures BTC arranco en 2019
PRETRAIN_EPOCHS  = 100      # mas pasadas → politica mas refinada

# Epsilon con el que el bot arranca en vivo tras un entrenamiento fresco.
# No queremos 0.05 (solo explotar) sino algo de exploracion para seguir aprendiendo.
EPSILON_LIVE_START = 0.20

# ── Live ──────────────────────────────────────────────────────────────────────
LOOKBACK          = 100     # velas a descargar en cada tick (>= WINDOW + margen)
CHECK_INTERVAL_S  = 3600    # segundos entre ticks (1h = 1 vela, igual que el entrenamiento)
TRAIN_EVERY_STEPS = 24      # entrenar cada N pasos en live (~1 dia)
SAVE_EVERY_STEPS  = 24      # guardar checkpoint cada N pasos (~1 dia con ticks de 1h)
