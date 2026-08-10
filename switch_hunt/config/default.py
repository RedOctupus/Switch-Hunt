"""游戏运行配置：光源、鬼移动、出生距离。"""

LIGHT_SYSTEM = {
    'initial_charges': 3,
    'max_charges': 3,
    'active_duration': 3.5,
    'cooldown_duration': 5.0,
    'radius': 3,
    # 强化光源：更大照射与定身威胁范围（格）
    'enhanced_radius': 5,
    'ambush_radius': 2,
    'stun_exposure_time': 1.0,
    'stun_duration': 2000,
    'auto_light_range': 3,
    'light_reaction_delay': 0.3,
}

# 鬼速度 = 玩家 1.2 倍（PLAYER_SPEED=150 / TILE_SIZE=32 → 5.625 格/秒）
GHOST_MOVE = {
    'speed': 5.625,
    'grid_size': 32,
}

GHOST_SPAWN = {
    'min_steps': 7,
    'max_steps': 12,
    'max_attempts': 50,
}
