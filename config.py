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
TIMEFRAME = "5m"

# ── Entorno de trading ────────────────────────────────────────────────────────
WINDOW        = 50          # velas que ve el agente como estado
INITIAL_CAP   = 1000.0      # capital inicial en USDT
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
PRETRAIN_CANDLES = 200_000  # ~2 años de velas 5m — cubre bull 2021, bear 2022, recovery 2023-2024
PRETRAIN_EPOCHS  = 100      # mas pasadas → politica mas refinada

# Epsilon con el que el bot arranca en vivo tras un entrenamiento fresco.
# No queremos 0.05 (solo explotar) sino algo de exploracion para seguir aprendiendo.
EPSILON_LIVE_START = 0.20

# ── Live ──────────────────────────────────────────────────────────────────────
LOOKBACK          = 200     # velas a descargar en cada tick
CHECK_INTERVAL_S  = 60      # segundos entre ticks
TRAIN_EVERY_STEPS = 50      # entrenar cada N pasos en live
SAVE_EVERY_STEPS  = 500     # guardar checkpoint cada N pasos
