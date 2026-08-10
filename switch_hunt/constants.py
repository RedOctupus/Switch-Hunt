"""全局常量：屏幕、地图、实体、颜色、DQN 维度。"""
SCREEN_WIDTH = 1024      # 屏幕宽度（像素）
SCREEN_HEIGHT = 768      # 屏幕高度（像素）
FPS = 60                 # 游戏帧率

# 地图设置
TILE_SIZE = 32           # 每个格子的大小（像素）
MAP_WIDTH = 21           # 地图宽度（格子数）- 使用奇数确保迷宫生成正常
MAP_HEIGHT = 21          # 地图高度（格子数）- 使用奇数确保迷宫生成正常

# 玩家设置
PLAYER_RADIUS = 12       # 玩家圆形碰撞箱半径（像素）
PLAYER_SPEED = 150       # 玩家移动速度（像素/秒）
PLAYER_MAX_ENERGY = 100  # 玩家最大能量值
PLAYER_ENERGY_DECAY = 10  # 强化光源能量消耗（点/秒）
PLAYER_ENERGY_REGEN = 0   # 修改：取消能量自动恢复，整局游戏固定100点

# 光源设置
LIGHT_RADIUS_NORMAL = 3   # 普通光源半径（格子数）
LIGHT_RADIUS_ENHANCED = 4 # 强化光源半径（格子数）
GHOST_FREEZE_DURATION = 3.0  # 强化光源定身鬼的持续时间（秒）

# 鬼设置
GHOST_SPEED_RATIO = 1.2   # 鬼速度是玩家的1.2倍
GHOST_RADIUS = 15         # 鬼的碰撞箱半径（像素）

# 宝藏设置
TREASURE_COUNT = 8        # 宝藏数量（随地图扩大增加）
TREASURE_ENERGY_RESTORE = 50  # 拾取宝藏恢复的能量值

# DQN预留参数
STATE_CHANNELS = 7  # V7: 7通道（含光源CD状态）        # v6.2: 增加到6通道（添加A*路径通道）
STATE_SIZE = 21           # 状态矩阵大小（21×21）
ACTION_SPACE = 4          # 动作空间大小（上、下、左、右）

# 颜色定义（兼容旧代码；局内精修色见 game.theme）
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (128, 128, 128)
COLOR_DARK_GRAY = (64, 64, 64)
COLOR_LIGHT_GRAY = (100, 100, 100)
COLOR_YELLOW = (255, 200, 80)
COLOR_BLUE = (70, 150, 255)
COLOR_GREEN = (72, 196, 140)
COLOR_RED = (220, 64, 72)
COLOR_ORANGE = (255, 160, 60)
COLOR_GOLD = (242, 201, 76)
COLOR_CYAN = (88, 196, 210)

# 墙壁颜色
COLOR_WALL = (58, 72, 92)
COLOR_WALL_BORDER = (78, 96, 118)
COLOR_FLOOR = (22, 26, 36)

# 鬼颜色
COLOR_GHOST_NORMAL = (210, 48, 58)
COLOR_GHOST_STUNNED = (90, 140, 255)
