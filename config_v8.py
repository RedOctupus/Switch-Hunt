"""
DQN V8 配置文件 - 三阶段博弈行为训练
更新日期: 2026-03-01
版本: V8.25
更新内容:
  - 鬼速度统一为玩家1.2倍（150*1.2/32=5.625格/s），删除sprint_speed区分
  - stun_radius→ambush_radius：2格重定义为高风险引诱区（非定身检测半径）
  - stun_exposure_time=1.0s：训练&游戏统一，鬼在enhanced光源内持续1秒才定身
  - light_reaction_delay=0.3s：玩家AI检测到鬼后300ms再开灯
  - GHOST_SPAWN：控制鬼出生距离玩家的A*步数范围（min_steps~max_steps）
"""

# ==================== DQN参数 ====================
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.98

LEARNING_RATE = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 50000
TARGET_UPDATE = 500

Q_VALUE_CLIP = (-30, 60)
GRAD_CLIP = 0.5

# ==================== 鬼奖励配置 (V8.23 三阶段博弈) ====================
GHOST_REWARD = {
    # 核心目标
    'catch_player': 30.0,

    # --- 诱饵模式（玩家有光源且光源未激活）---
    'correct_direction_stalk': 0.5,
    'wrong_direction_stalk': -0.5,
    'approach_bonus_stalk': 0.3,
    'retreat_penalty_stalk': -0.1,
    'in_trigger_zone': 1.0,             # 进入3格引诱区（触发玩家开灯）
    'ambush_bonus': 2.5,                # V8.25: 2格高风险引诱区奖励（非定身范围）

    # 引诱成功（玩家开灯 + 鬼未被定身）
    'bait_success': 15.0,

    # --- 光源激活中（危险，需要撤退）---
    'retreat_bonus_light': 0.3,
    'approach_penalty_light': -0.2,
    'stunned_penalty': -8.0,

    # --- 冲刺模式（玩家光源耗尽）---
    'correct_direction_sprint': 2.0,
    'wrong_direction_sprint': -2.0,
    'approach_bonus_sprint': 0.5,

    # 通用
    'wall_hit': -0.5,
}

# ==================== 光源系统 (V8.25) ====================
LIGHT_SYSTEM = {
    'initial_charges': 3,
    'max_charges': 3,
    'active_duration': 3.5,         # 光源激活持续时间（秒）
    'cooldown_duration': 5.0,       # 冷却时间（秒）
    'radius': 3,                    # 普通光源/视野半径（格）
    'enhanced_radius': 3,           # 激活光源半径（格）— 定身检测使用此范围
    # V8.25: stun_radius→ambush_radius（2格高风险引诱区，不再是定身检测半径）
    'ambush_radius': 2,             # 高风险引诱区半径（获得ambush_bonus奖励）
    # V8.25: 统一定身曝光时间（训练和游戏相同规则）
    'stun_exposure_time': 1.0,      # 在enhanced光源内持续多久才定身（秒）
    'stun_duration': 2000,          # 定身持续时间（毫秒）
    'auto_light_range': 3,          # 玩家AI检测鬼的范围（格），等于视野半径
    # V8.25: 玩家AI检测到鬼后等待300ms再开灯（模拟真实反应时间）
    'light_reaction_delay': 0.3,    # 玩家AI开灯反应延迟（秒）
}

# ==================== 鬼移动配置 (V8.25) ====================
# V8.25: 速度统一为玩家1.2倍（PLAYER_SPEED=150, GHOST_SPEED_RATIO=1.2, TILE_SIZE=32）
# 150 * 1.2 / 32 = 5.625 格/秒。删除sprint_speed，sprint只是奖励结构区分，速度不变。
GHOST_MOVE = {
    'speed': 5.625,    # 恒等于 PLAYER_SPEED * GHOST_SPEED_RATIO / TILE_SIZE
    'grid_size': 32,
}

# ==================== 训练参数 ====================
MAX_EPISODE_STEPS = 4000     # 保留兼容（旧训练系统）
MAX_DISCRETE_STEPS = 300     # 离散格间步数上限（每集最多300次格间移动）
PATH_UPDATE_INTERVAL = 5     # A*路径更新间隔（步）

# ==================== 课程学习配置 (V8.23) ====================
CURRICULUM = {
    'phase1_end': 100,    # EP   1-100: Phase 1 - 纯冲刺（静止玩家，0次光源）
    'phase2_end': 250,    # EP 101-250: Phase 2 - 引诱学习（静止玩家，3次光源，300ms延迟开灯）
    'phase3_end': 500,    # EP 251-500: Phase 3 - 完整博弈（AI玩家移动，3次光源）
}

# ==================== 鬼出生位置配置 (V8.25新增) ====================
GHOST_SPAWN = {
    'min_steps': 7,       # 出生距离玩家的最少A*步数（太近易被玩家引诱）
    'max_steps': 12,      # 出生距离玩家的最多A*步数（太远训练效率低）
    'max_attempts': 50,   # 寻找有效位置的最大尝试次数（失败则随机放置）
}

# ==================== 检查点配置 (V8.22) ====================
CHECKPOINT_CONFIG = {
    'enabled': True,
    'interval': 10,
    'max_keep': 5,
    'save_dir': 'checkpoints',
    'save_stats': True,
}

# ==================== 玩家AI配置 (V8.25) ====================
PLAYER_AI = {
    'enabled_in_training': True,    # Phase 3启用玩家AI移动
    'light_range': 3,               # 鬼进入3格范围时考虑开灯（与视野一致）
    'path_update_interval': 2.0,    # 寻路更新间隔（秒）
    'stuck_threshold': 0.3,         # 卡住检测阈值（秒）
    'training_light_charges': 3,
    'training_light_reset_count': 999,
}
