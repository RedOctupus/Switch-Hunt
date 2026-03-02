# 开关猎杀 (Switch Hunt) DQN V8 - AI Agent Guide

## Project Overview

**开关猎杀 (Switch Hunt)** 是一个基于Pygame的2D迷宫捉迷藏游戏，结合深度强化学习(DQN)训练鬼AI追捕玩家。本项目是V8版本，采用网格对齐移动系统和三阶段博弈行为学习。

### 核心特性
- **网格对齐移动**: 鬼球直径32px(恰好一个格子)，圆心始终对齐格子中心，移动耗时0.5秒
- **方向趋势奖励**: 不关注"在哪里"，只关注"往哪走"，根据A*路径方向给予奖励
- **三阶段课程学习**: Phase1纯冲刺 → Phase2引诱学习 → Phase3完整博弈
- **7通道状态编码**: 墙壁、鬼位置、玩家位置、A*路径、危险区、CD状态、光源次数

## Technology Stack

| 组件 | 技术 |
|------|------|
| 游戏引擎 | Pygame |
| 深度学习 | PyTorch |
| 数值计算 | NumPy |
| 可视化 | Matplotlib |
| 语言 | Python 3.x |

## Project Structure

```
DQN_Final/
├── switch_hunt_v8_game.py      # V8游戏本体（完整功能，含渲染、音效、UI）
├── dqn_training_system_v8.py   # V8训练系统（DQN环境、训练循环）
├── dqn_model_v8.py             # V8 DQN模型定义（网络架构、ReplayBuffer、DQNAI）
├── config_v8.py                # V8配置文件（所有可调参数）
├── switch_hunt_v7_base.py      # V7基础库（共享常量、枚举、辅助函数）
├── maze_system.py              # 迷宫生成系统（DFS算法、碰撞检测）
├── train.py                    # 快速训练入口脚本（推荐）
├── verify_v8.py                # V8系统验证脚本
├── models/                     # 训练好的模型保存目录
│   └── ghost_v8.pth           # 鬼AI模型
├── checkpoints/                # 训练检查点
├── logs/                       # 训练日志和可视化图表
├── README_v8.md                # V8功能说明文档
├── CHANGELOG_V8.md             # V8变更日志
└── ITERATION_LOG.md            # 迭代记录
```

## Key Files Description

### 1. config_v8.py - 配置中心
所有可调参数集中在此文件：
- `GHOST_REWARD`: 鬼奖励配置（三阶段博弈）
- `LIGHT_SYSTEM`: 光源系统参数
- `GHOST_MOVE`: 鬼移动配置
- `CURRICULUM`: 课程学习阶段划分
- `CHECKPOINT_CONFIG`: 检查点配置
- `PLAYER_AI`: 玩家AI配置

### 2. dqn_model_v8.py - DQN模型
- `DQN`: 神经网络类（7通道输入，4动作输出）
- `ReplayBuffer`: 经验回放缓冲区
- `DQNAI`: DQN代理类（训练/推理接口）
- `ConfigurableDQNAI`: 支持配置文件的DQNAI

网络架构:
```
Conv2d(7→32) → BN → ReLU
Conv2d(32→64) → BN → ReLU  
Conv2d(64→64) → BN → ReLU
Flatten → FC(28224→256) → ReLU → Dropout
FC(256→128) → ReLU → Dropout
FC(128→4)
```

### 3. switch_hunt_v8_game.py - 游戏本体
主要类:
- `GameV8`: 游戏主类
- `PlayerV8`: 玩家类（支持AI控制）
- `DQNGhostV8`: DQN控制的鬼类（网格对齐移动）
- `SoundManager`: 程序化音效管理器（无需外部音频文件）
- `UISystem`: UI系统

### 4. dqn_training_system_v8.py - 训练系统
主要类/函数:
- `CheckpointManager`: 检查点管理（保存/加载/清理）
- `SwitchHuntTrainingEnvV8`: 训练环境（V8核心）
- `train_ghost_v8()`: 主训练函数（三阶段课程学习）
- `step_discrete()`: 离散化训练步骤（V8.23关键修复）

## Build and Run Commands

### 安装依赖
项目没有requirements.txt，需要手动安装:
```bash
pip install pygame numpy torch matplotlib
```

### 运行完整游戏（带训练好的模型）
```bash
python switch_hunt_v8_game.py
```

### 训练DQN（推荐方式）
```bash
# 默认训练500回合，无渲染
python train.py

# 训练1000回合
python train.py -e 1000

# 带实时渲染训练
python train.py -r 1

# 完整参数
python train.py -e 500 -r 0 -p 10
```

### 直接运行训练系统
```bash
# 默认500回合，带渲染（与train.py不同）
python dqn_training_system_v8.py

# 无渲染
python dqn_training_system_v8.py --render 0

# 1000回合
python dqn_training_system_v8.py -e 1000 -r 0
```

### 验证系统
```bash
python verify_v8.py
```

### 测试模型
```bash
python dqn_model_v8.py  # 测试网络架构
```

## Game Controls

| 按键 | 功能 |
|------|------|
| WASD/方向键 | 移动玩家 |
| 空格 | 开启强化光源 |
| F1 | 切换作弊模式（显示全地图） |
| F2 | 切换AI演示模式（玩家自动寻宝） |
| F3 | 显示/隐藏鬼的A*路径 |
| P | 暂停游戏 |
| ESC | 返回菜单 |

## Training System Architecture

### 三阶段课程学习 (V8.23)

| 阶段 | 回合范围 | 玩家状态 | 光源次数 | 行为目标 |
|------|----------|----------|----------|----------|
| Phase1 | EP 1-100 | 静止 | 0 | 纯冲刺方向学习 |
| Phase2 | EP 101-250 | 静止 | 3 | 引诱机制学习（300ms延迟开灯） |
| Phase3 | EP 251-500 | AI移动 | 3 | 完整博弈 |

### 奖励体系 (V8.25)

| 奖励键 | 值 | 说明 |
|--------|-----|------|
| `catch_player` | +30.0 | 抓到玩家 |
| `bait_success` | +15.0 | 引诱成功（玩家开灯+鬼未被定身） |
| `ambush_bonus` | +2.5 | 进入2格高风险引诱区 |
| `in_trigger_zone` | +1.0 | 进入3格引诱区 |
| `stunned_penalty` | -8.0 | 被定身 |

### 训练指标

训练时关注以下指标:
- **Correct%**: 正确方向率（目标>70%）
- **Catch Rate**: 抓取成功率（目标>80%）
- **Avg Reward**: 平均回合奖励
- **Bait Success**: 引诱成功次数

### 状态编码 (7通道 21×21)

```python
通道0: 墙壁地图
通道1: 鬼位置 (one-hot)
通道2: 玩家位置 (one-hot)
通道3: A*路径
通道4: 危险区（enhanced_radius=3格，光源激活时）
通道5: 光源CD状态（CD中=1.0）
通道6: 玩家光源次数归一化比例（charges/max_charges）
```

## Code Conventions

### 版本标签
- 代码中使用 `[V8]`、`[V8.23]`、`[V8.25]` 等标签标记版本特定代码
- 关键修复使用 `BUG-xxx` 标记在CHANGELOG中

### 命名规范
- 类名: `PascalCase` (e.g., `DQNGhostV8`, `SwitchHuntTrainingEnvV8`)
- 函数/方法: `snake_case` (e.g., `step_discrete`, `calculate_reward`)
- 常量: `UPPER_CASE` (e.g., `EPSILON_START`, `GHOST_REWARD`)
- 私有方法: `_leading_underscore` (e.g., `_check_light_stun`)

### 文档字符串
使用中文文档字符串，包含:
- 功能描述
- 参数说明（类型、含义）
- 返回值说明
- 版本修改记录

示例:
```python
def step_discrete(self, action):
    """V8.23: 离散化训练步骤 — 1次调用 = 1次完整格间移动
    
    核心改进（修复Bug1）:
    原step_train_ghost每帧调用，15帧中14帧忽略action但仍计算奖励。
    现在每次完整执行一次格间移动，确保 action → 实际执行 → 奖励 的因果链正确。
    
    Returns: next_state, reward, done, info
    """
```

### 配置管理
- 所有可调参数集中在 `config_v8.py`
- 训练系统通过 `from config_v8 import ...` 导入
- 游戏逻辑通过传参或全局配置访问

## Testing Strategy

### 验证脚本 (verify_v8.py)
包含6个测试:
1. `test_imports`: 模块导入测试
2. `test_game_init`: 游戏初始化测试
3. `test_ghost_grid_alignment`: 网格对齐测试
4. `test_ghost_movement`: 鬼移动系统测试
5. `test_direction_reward`: 方向奖励计算测试
6. `test_state_encoding`: 状态编码测试

运行: `python verify_v8.py`

### 模型测试 (dqn_model_v8.py)
测试网络架构:
- 输入/输出形状
- 前向传播
- 动作选择

运行: `python dqn_model_v8.py`

### 调试模式
训练时设置环境变量显示调试信息:
```bash
set DQN_TRAINING=1
python dqn_training_system_v8.py
```

## Checkpoint System

训练过程中自动保存检查点:
- 保存位置: `checkpoints/`
- 命名格式: `checkpoint_ep{轮数}_{时间戳}.pth`
- 统计文件: `stats_ep{轮数}_{时间戳}.json`
- 默认每10轮保存，保留最近5个

### 检查点配置
```python
CHECKPOINT_CONFIG = {
    'enabled': True,
    'interval': 10,
    'max_keep': 5,
    'save_dir': 'checkpoints',
    'save_stats': True,
}
```

## Common Issues and Fixes

### 1. 内存泄漏修复 (V8.23)
```python
# 修复前
self.losses = []  # 无界列表

# 修复后
self.losses = deque(maxlen=50000)  # 有界队列
```

### 2. Dropout/BN不确定性修复 (V8.23)
```python
# 修复前
q_value = self.Qnet(state)  # train()模式下有随机性

# 修复后
self.Qnet.eval()
with torch.no_grad():
    q_value = self.Qnet(state)
self.Qnet.train()
```

### 3. 离散化训练步骤 (V8.23)
```python
# 修复前: step_train_ghost每帧调用，奖励计算错误

# 修复后: step_discrete每次完成完整格间移动
while ghost.is_moving:
    ghost._continue_move(dt)
    # ... 物理推进
# 然后计算奖励
```

## Development Workflow

1. **修改配置**: 编辑 `config_v8.py`
2. **修改模型**: 编辑 `dqn_model_v8.py`
3. **修改游戏**: 编辑 `switch_hunt_v8_game.py`
4. **修改训练**: 编辑 `dqn_training_system_v8.py`
5. **验证**: 运行 `python verify_v8.py`
6. **训练**: 运行 `python train.py`
7. **测试游戏**: 运行 `python switch_hunt_v8_game.py`

## File Dependencies

```
switch_hunt_v8_game.py
    ├── switch_hunt_v7_base.py
    └── config_v8.py

dqn_training_system_v8.py
    ├── switch_hunt_v8_game.py
    ├── dqn_model_v8.py
    └── config_v8.py

dqn_model_v8.py
    └── (PyTorch only)

train.py
    └── dqn_training_system_v8.py

verify_v8.py
    ├── switch_hunt_v8_game.py
    ├── dqn_training_system_v8.py
    ├── dqn_model_v8.py
    └── config_v8.py
```

## Notes for AI Agents

1. **不要修改测试文件**: `verify_v8.py` 是验证工具，保持原样
2. **配置优先**: 优先修改 `config_v8.py` 而非硬编码参数
3. **版本标签**: 新增代码添加版本标签（如 `[V8.26]`）
4. **中文注释**: 保持中文注释和文档字符串
5. **检查点兼容**: 修改模型结构时考虑检查点兼容性
6. **离散化步骤**: 训练相关修改应在 `step_discrete()` 中进行
