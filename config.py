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
LR               = 1e-4
GAMMA            = 0.99     # descuento de recompensas futuras
EPSILON_START    = 1.0      # exploracion inicial (100% aleatoria)
EPSILON_MIN      = 0.05     # exploracion minima (5%)
EPSILON_DECAY    = 0.9995   # decaimiento por paso
BUFFER_SIZE      = 100_000  # tamano del replay buffer
BATCH_SIZE       = 128
TARGET_UPDATE    = 500      # pasos entre actualizacion de red objetivo

# ── Entrenamiento offline ─────────────────────────────────────────────────────
PRETRAIN_CANDLES = 5000     # velas historicas para pre-entrenar
PRETRAIN_EPOCHS  = 30       # pasadas sobre datos historicos

# ── Live ──────────────────────────────────────────────────────────────────────
LOOKBACK          = 200     # velas a descargar en cada tick
CHECK_INTERVAL_S  = 60      # segundos entre ticks
TRAIN_EVERY_STEPS = 50      # entrenar cada N pasos en live
SAVE_EVERY_STEPS  = 500     # guardar checkpoint cada N pasos
