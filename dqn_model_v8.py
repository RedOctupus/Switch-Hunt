"""
DQN模型 v8.23 - 三阶段博弈行为系统
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


class DQN(nn.Module):
    """
    DQN网络 v8.23
    - 输入: 7通道 (墙壁 + 鬼位置 + 玩家位置 + A*路径 + 危险区 + CD状态 + 光源归一化次数)
    """
    def __init__(self, channels=7, height=21, width=21, action_size=4):
        super().__init__()
        # V7: 输入通道改为7
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.feature_size = 64 * height * width

        self.fc1 = nn.Linear(self.feature_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.stack(state),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float),
            torch.stack(next_state),
            torch.tensor(done, dtype=torch.float)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAI:
    """
    DQN Agent v8.15
    - 支持7通道状态输入（墙壁, 鬼位置, 玩家位置, A*路径, 光源范围, CD状态, 是否有光源）
    """
    def __init__(self,
                 state_channels=7,  # V8.15: 7通道
                 state_size=21,
                 action_size=4,
                 lr=0.0005,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.99,      # V8.21: 加快衰减
                 epsilon_min=0.05,
                 grad_clip=0.5,           # V8.21: 放宽梯度裁剪
                 target_update=500,       # V8.21: 降低更新频率
                 q_value_clip=(-50.0, 50.0)):  # V8.21: 放宽Q值裁剪
        
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.grad_clip = grad_clip
        self.target_update = target_update
        self.q_value_clip = q_value_clip

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 只在非训练模式显示设备信息（避免训练时重复输出）
        if os.environ.get('DQN_TRAINING') != '1':
            print(f"[V8] Using device: {self.device}")
            if torch.cuda.is_available():
                print(f"[V8] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[V8] State channels: {state_channels} (walls, ghost_pos, player_pos, astar_path, light_range, cd_state, has_charges)")

        # V7: Q网络接受3通道输入（简化版）
        self.Qnet = DQN(state_channels, state_size, state_size, action_size).to(self.device)
        self.Tnet = DQN(state_channels, state_size, state_size, action_size).to(self.device)
        self.Tnet.load_state_dict(self.Qnet.state_dict())
        self.Tnet.eval()

        self.optimizer = optim.Adam(self.Qnet.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        self.memory = ReplayBuffer(capacity=50000)

        self.batch_size = 64
        self.step_count = 0
        self.soft_update_tau = 0.005  # 增加软更新速率（从0.001到0.005），Target网络更新更快

        self.losses = deque(maxlen=50000)   # V8.23: 防止内存泄漏
        self.q_values = deque(maxlen=50000)  # V8.23: 防止内存泄漏

    def get_action(self, state, training=True):
        """选择动作 - V8.23: 推理时切换eval模式（修复Dropout/BN不确定性）"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        self.Qnet.eval()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_value = self.Qnet(state)
            self.q_values.append(q_value.max().item())
            action = q_value.argmax().item()
        self.Qnet.train()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """学习更新"""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 当前Q值
        current_q = self.Qnet(states).gather(1, actions.unsqueeze(1))

        # 目标Q值 - Double DQN（减少过度估计）
        with torch.no_grad():
            # Double DQN: 用Qnet选择动作，用Tnet评估动作
            next_actions = self.Qnet(next_states).argmax(1)
            next_q = self.Tnet(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
            # Q值裁剪，防止爆炸
            target_q = torch.clamp(target_q, self.q_value_clip[0], self.q_value_clip[1])

        # Huber Loss
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Qnet.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        self.step_count += 1
        
        # 记录loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        # 更新Target Network - 使用软更新（Polyak平均），更平滑稳定
        self._soft_update_target_network()
        
        # 每10000步报告一次
        if self.step_count % 10000 == 0:
            print(f"[V7 Step {self.step_count}] Target network soft-updated (tau={self.soft_update_tau})")

        return loss_value

    def _soft_update_target_network(self):
        """软更新Target Network（Polyak平均）
        Tnet = tau * Qnet + (1 - tau) * Tnet
        这样更新更平滑，训练更稳定
        """
        for target_param, policy_param in zip(self.Tnet.parameters(), self.Qnet.parameters()):
            target_param.data.copy_(
                self.soft_update_tau * policy_param.data + (1.0 - self.soft_update_tau) * target_param.data
            )

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_learning_rate(self):
        """更新学习率"""
        self.scheduler.step()

    def get_stats(self):
        """获取训练统计"""
        # V8.23修复: deque不支持负索引切片，需先转list
        recent_losses = list(self.losses)[-100:]
        recent_qvals = list(self.q_values)[-100:]
        stats = {
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'avg_loss': sum(recent_losses) / len(recent_losses) if recent_losses else 0,
            'avg_q_value': sum(recent_qvals) / len(recent_qvals) if recent_qvals else 0
        }
        return stats

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_net': self.Qnet.state_dict(),
            'target_net': self.Tnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, path)
        print(f"[V8] Model saved to {path}")

    def load(self, path):
        """加载模型 - 兼容检查
        
        Raises:
            ValueError: 当模型通道数与当前网络不匹配时
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # 检查通道数兼容性 - V8: 支持6通道
        policy_state = checkpoint['policy_net']
        conv1_weight = policy_state.get('conv1.weight', policy_state.get('Qnet.conv1.weight'))
        if conv1_weight is not None:
            input_channels = conv1_weight.shape[1]  # [out, in, h, w]
            # 获取当前网络的输入通道数
            expected_channels = self.Qnet.conv1.in_channels
            if input_channels != expected_channels:
                # V8.20: 抛出异常而不是静默返回，让调用者知道加载失败
                raise ValueError(f"Model has {input_channels} channels, expected {expected_channels}")
        
        self.Qnet.load_state_dict(checkpoint['policy_net'])
        self.Tnet.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        if os.environ.get('DQN_TRAINING') != '1':
            print(f"[V8] Model loaded from {path}")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing DQN v8.23 Network (7 channels)...")
    print("=" * 70)

    # 测试7通道输入
    batch_size = 4
    state = torch.randn(batch_size, 7, 21, 21)  # V8.23: 7通道

    # 测试DQN
    dqn = DQN(channels=7, height=21, width=21, action_size=4)
    output = dqn(state)
    print(f"Input shape: {state.shape}")
    print(f"DQN output shape: {output.shape}")  # 应该是 (4, 4)

    # 测试Agent (7通道)
    agent = DQNAI(state_channels=7, state_size=21, action_size=4)
    test_state = torch.randn(7, 21, 21).numpy()
    action = agent.get_action(test_state, training=False)
    print(f"Selected action: {action}")

    print("\n[V8] All tests passed!")
    print("=" * 70)


# ==================== ConfigurableDQNAI (V8) ====================

class ConfigurableDQNAI(DQNAI):
    """支持配置文件的DQNAI (V8)"""
    
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = {}
        
        # V8.15: 7通道状态输入（墙壁, 鬼位置, 玩家位置, A*路径, 光源范围, CD状态, 是否有光源）
        super().__init__(
            state_channels=config.get('state_channels', 7),
            state_size=config.get('state_size', 21),
            action_size=config.get('action_size', 4),
            lr=config.get('lr', 0.001),
            gamma=config.get('gamma', 0.99),
            epsilon=config.get('epsilon_start', 1.0),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            epsilon_min=config.get('epsilon_min', 0.05),
            grad_clip=config.get('grad_clip', 0.1),
            target_update=config.get('target_update', 100),
            q_value_clip=config.get('q_value_clip', (-10.0, 10.0))
        )
        
        self.memory = ReplayBuffer(capacity=config.get('buffer_size', 50000))
        self.batch_size = config.get('batch_size', 64)
        self.decay_schedule = config.get('decay_schedule', 'exponential')
        self.total_episodes = config.get('total_episodes', 500)
        self.current_episode = 0
        
        # 保存epsilon_start用于decay_epsilon
        self.epsilon_start = config.get('epsilon_start', 1.0)
    
    def set_episode(self, episode):
        self.current_episode = episode
    
    def decay_epsilon(self):
        if self.decay_schedule == 'step':
            # 简化的step衰减
            progress = self.current_episode / self.total_episodes
            self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * max(0, 1 - progress)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
