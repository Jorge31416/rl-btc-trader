# RL BTC Trader — Agente de Reinforcement Learning

Bot de trading de BTC que aprende a invertir por refuerzo, igual que los bots que aprenden a jugar Space Invaders: empieza sin saber nada, prueba acciones, recibe recompensas por sus aciertos y va mejorando solo.

## Como funciona

```
Estado (lo que ve)         Accion (lo que decide)      Recompensa (lo que aprende)
──────────────────         ──────────────────────       ───────────────────────────
Ultimas 50 velas           HOLD  — no hacer nada        PnL del trade cerrado
  · retornos               LONG  — comprar               + si gana
  · high/low/open          SHORT — vender                - si pierde
  · volumen                CLOSE — cerrar posicion
  · posicion actual
```

No le decimos que mire RSI, EMA ni ningun indicador. La red neuronal descubre sola que patrones de velas llevan a beneficio.

## Arquitectura

```
rl_btc_trader/
├── env/trading_env.py   # Entorno Gym: estado, acciones, recompensas
├── agent/dqn.py         # Red neuronal DQN + replay buffer
├── train.py             # Pre-entrenamiento offline con datos historicos
└── live.py              # Bot en vivo con aprendizaje continuo
```

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Crea el fichero `.env` (copia `.env.example`):
```
BINANCE_API_KEY=tu_clave_demo
BINANCE_API_SECRET=tu_secreto_demo
TELEGRAM_TOKEN=opcional
TELEGRAM_CHAT_ID=opcional
```

## Uso

### 1. Pre-entrenar (recomendado)
```bash
python train.py
```
El agente juega miles de episodios sobre datos historicos hasta bajar epsilon.

### 2. Lanzar en vivo
```bash
python live.py
```
Opera en Binance Demo Trading y sigue aprendiendo de cada trade real.

## Diferencia con GA

| GA (ga_btc_evolver)              | RL (este proyecto)               |
|----------------------------------|----------------------------------|
| Evoluciona parametros de reglas  | Aprende comportamiento desde cero|
| RSI, EMA, MACD predefinidos      | Solo velas crudas normalizadas   |
| Fitness por Sharpe ratio         | Recompensa por PnL directo       |
| No mejora dentro de un individuo | Mejora continuamente con cada trade|
