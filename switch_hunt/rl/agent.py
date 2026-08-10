"""DQN 推理 Agent（仅加载权重与选动作，不含训练）。"""
from __future__ import annotations

import torch

from switch_hunt.rl.network import DQN


class DQNAI:
    """游戏用鬼 AI：加载 ghost_v8.pth 后推理。"""

    def __init__(
        self,
        state_channels: int = 7,
        state_size: int = 21,
        action_size: int = 4,
        epsilon: float = 0.0,
        **_ignored,
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Qnet = DQN(state_channels, state_size, state_size, action_size).to(self.device)
        self.Qnet.eval()
        print(f"[V8] Using device: {self.device}")

    def get_action(self, state, training: bool = False):
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return int(self.Qnet(tensor).argmax().item())

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
            policy_state = checkpoint["policy_net"]
        else:
            policy_state = checkpoint

        weight = policy_state.get("conv1.weight")
        if weight is not None and weight.shape[1] != self.Qnet.conv1.in_channels:
            raise ValueError(
                f"Model has {weight.shape[1]} channels, expected {self.Qnet.conv1.in_channels}"
            )

        self.Qnet.load_state_dict(policy_state)
        self.Qnet.eval()
        if isinstance(checkpoint, dict) and "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        print(f"[V8] Model loaded from {path}")
