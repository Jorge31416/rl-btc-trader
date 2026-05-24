"""
Agente DQN (Deep Q-Network) — v2 con Dueling DQN.

Mejoras respecto a v1:
  - Dueling DQN: separa valor de estado V(s) y ventaja de accion A(s,a)
  - Dropout(0.1): evita overfitting a periodos especificos del mercado
  - CircularBuffer con lista: O(1) indexing → muestreo 10x mas rapido que deque
  - decay_epsilon=False en train_step para que train.py controle epsilon por epoca
"""
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config

log = logging.getLogger(__name__)


# ── Red neuronal (Dueling DQN) ────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    V(s) = valor de estar en el estado s (independiente de la accion)
    A(s,a) = ventaja relativa de la accion a en el estado s
    """
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(state_size)
        self.fc1  = nn.Linear(state_size, 256)
        self.fc2  = nn.Linear(256, 256)
        self.fc3  = nn.Linear(256, 128)
        self.act  = nn.LeakyReLU(0.01)
        self.drop = nn.Dropout(0.1)

        # Stream de valor V(s)
        self.val_fc  = nn.Linear(128, 64)
        self.val_out = nn.Linear(64, 1)

        # Stream de ventaja A(s,a)
        self.adv_fc  = nn.Linear(128, 64)
        self.adv_out = nn.Linear(64, action_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        for out in [self.val_out, self.adv_out]:
            nn.init.orthogonal_(out.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x)) + x    # skip connection
        x = self.drop(self.act(self.fc3(x)))

        val = self.val_out(self.act(self.val_fc(x)))   # [B, 1]
        adv = self.adv_out(self.act(self.adv_fc(x)))   # [B, action_size]

        return val + (adv - adv.mean(dim=1, keepdim=True))


# ── Replay buffer (circular, O(1) indexing) ───────────────────────────────────

class ReplayBuffer:
    """
    Buffer circular basado en lista Python.
    random.sample sobre lista es O(1) por acceso → mucho mas rapido que deque
    que requiere O(n) para indexar elementos al azar.
    """

    def __init__(self, capacity: int = config.BUFFER_SIZE):
        self.capacity = capacity
        self._buf: list = []
        self._pos: int  = 0

    def push(self, state, action, reward, next_state, done):
        exp = (
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        )
        if len(self._buf) < self.capacity:
            self._buf.append(exp)
        else:
            self._buf[self._pos] = exp
            self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r),
                np.array(ns), np.array(d))

    def get_all(self) -> list:
        """Devuelve todas las experiencias para persistencia en disco."""
        return list(self._buf)

    def load_from(self, experiences: list):
        """Carga experiencias guardadas en disco."""
        self._buf = list(experiences)[-self.capacity:]
        self._pos = len(self._buf) % self.capacity

    def __len__(self):
        return len(self._buf)


# ── Agente DQN ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Agente que aprende que hacer en cada estado del mercado.
    """

    ACTION_NAMES = ["FLAT", "LONG", "SHORT"]

    def __init__(self, state_size: int, action_size: int = 3):
        self.state_size  = state_size
        self.action_size = action_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        log.info(f"DQN usando: {self.device}")

        self.q_net      = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=config.LR, eps=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20_000, gamma=0.5
        )
        self.loss_fn = nn.HuberLoss(delta=1.0)
        self.buffer  = ReplayBuffer()

        self.epsilon = config.EPSILON_START
        self.steps   = 0

    # ── Politica ──────────────────────────────────────────────────────────────

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        self.q_net.eval()
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(s)
        self.q_net.train()
        return int(q.argmax().item())

    # ── Aprendizaje ───────────────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self, decay_epsilon: bool = True) -> float | None:
        """
        Un paso de entrenamiento.
        decay_epsilon=False → el caller controla epsilon (entrenamiento offline).
        decay_epsilon=True  → decaimiento automatico (trading en vivo).
        """
        if len(self.buffer) < config.BATCH_SIZE:
            return None

        s, a, r, ns, d = self.buffer.sample(config.BATCH_SIZE)

        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        self.q_net.train()

        q_curr = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            self.target_net.eval()
            best_a   = self.q_net(ns).argmax(1)
            q_next   = self.target_net(ns).gather(1, best_a.unsqueeze(1)).squeeze(1)
            q_target = r + config.GAMMA * q_next * (1 - d)

        loss = self.loss_fn(q_curr, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        if decay_epsilon:
            self.epsilon = max(config.EPSILON_MIN,
                               self.epsilon * config.EPSILON_DECAY)

        self.steps += 1
        if self.steps % config.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

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
        log.info(f"Checkpoint guardado: {path} (eps={self.epsilon:.3f}, steps={self.steps:,})")

    def load(self, path: str = "checkpoints/dqn.pth"):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        log.info(f"Checkpoint cargado: {path} (eps={self.epsilon:.3f}, steps={self.steps:,})")
