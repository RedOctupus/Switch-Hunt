"""游戏状态与光源相关枚举。"""
from enum import Enum

class GameState(Enum):
    """游戏状态枚举"""
    MENU = "menu"       # 主菜单
    PLAYING = "playing" # 游戏进行中
    PAUSED = "paused"   # 暂停
    VICTORY = "victory" # 胜利
    GAME_OVER = "game_over"  # 失败


class LightMode(Enum):
    """光源模式枚举"""
    NORMAL = "normal"     # 普通光源
    ENHANCED = "enhanced" # 强化光源


class LightState(Enum):
    """V7: 光源状态枚举"""
    IDLE = "idle"         # 闲置，可开启
    ACTIVE = "active"     # 激活中（5秒）
    COOLDOWN = "cooldown" # 冷却中（3秒）


class GhostState(Enum):
    """鬼状态枚举"""
    NORMAL = "normal"   # 正常状态
    STUNNED = "stunned" # 定身状态
