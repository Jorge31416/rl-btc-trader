"""
Agente DQN (Deep Q-Network).

Mismo principio que los bots que aprenden a jugar Atari:
  - Red neuronal mapea estado -> valor de cada accion
  - Replay buffer almacena experiencias pasadas
  - Red objetivo (target) estabiliza el entrenamiento
  - Epsilon-greedy: explora aleatoriamente al principio,
    explota lo aprendido a medida que epsilon decrece
"""
import random
import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config

log = logging.getLogger(__name__)


# ── Red neuronal ──────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Recibe el vector de estado y devuelve el Q-valor de cada accion.
    Arquitectura: MLP con capas residuales para mejor gradiente.
    """
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(state_size)
        self.fc1  = nn.Linear(state_size, 256)
        self.fc2  = nn.Linear(256, 256)
        self.fc3  = nn.Linear(256, 128)
        self.out  = nn.Linear(128, action_size)
        self.act  = nn.LeakyReLU(0.01)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.out.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x)) + x     # skip connection
        x = self.act(self.fc3(x))
        return self.out(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Almacena experiencias (s, a, r, s', done) para entrenamiento offline."""

    def __init__(self, capacity: int = config.BUFFER_SIZE):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r),
                np.array(ns), np.array(d))

    def __len__(self):
        return len(self.buf)


# ── Agente DQN ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Agente que aprende que hacer en cada estado del mercado.
    No sabe nada de RSI, EMA ni ningun indicador — aprende solo
    mirando velas crudas y recibiendo recompensas por sus trades.
    """

    ACTION_NAMES = ["FLAT", "LONG", "SHORT"]

    def __init__(self, state_size: int, action_size: int = 3):
        self.state_size  = state_size
        self.action_size = action_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        log.info(f"DQN usando: {self.device}")

        # Red principal + red objetivo (para estabilidad del entrenamiento)
        self.q_net      = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=config.LR
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10000, gamma=0.5
        )
        self.loss_fn = nn.HuberLoss()
        self.buffer  = ReplayBuffer()

        # Exploracion epsilon-greedy
        self.epsilon = config.EPSILON_START
        self.steps   = 0

    # ── Politica ──────────────────────────────────────────────────────────────

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Elige accion.
        training=True  → epsilon-greedy (explora)
        training=False → greedy puro (explota)
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(s)
        return int(q.argmax().item())

    # ── Aprendizaje ───────────────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """Un paso de entrenamiento sobre un batch del replay buffer."""
        if len(self.buffer) < config.BATCH_SIZE:
            return None

        s, a, r, ns, d = self.buffer.sample(config.BATCH_SIZE)

        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        # Q actual
        q_curr = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Q objetivo (Double DQN: accion elegida por q_net, valor por target_net)
        with torch.no_grad():
            best_a   = self.q_net(ns).argmax(1)
            q_next   = self.target_net(ns).gather(1, best_a.unsqueeze(1)).squeeze(1)
            q_target = r + config.GAMMA * q_next * (1 - d)

        loss = self.loss_fn(q_curr, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Decaer epsilon
        self.epsilon = max(
            config.EPSILON_MIN,
            self.epsilon * config.EPSILON_DECAY
        )

        # Actualizar red objetivo periodicamente
        self.steps += 1
        if self.steps % config.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            log.debug(f"Target network actualizada (step {self.steps})")

        return float(loss.item())

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(self, path: str = "checkpoints/dqn.pth"):
        Path(path).parent.mkdir(exist_ok=True)
        torch.save({
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "steps":      self.steps,
        }, path)
        log.info(f"Checkpoint guardado: {path} (eps={self.epsilon:.3f})")

    def load(self, path: str = "checkpoints/dqn.pth"):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        log.info(f"Checkpoint cargado: {path} (eps={self.epsilon:.3f}, steps={self.steps})")
