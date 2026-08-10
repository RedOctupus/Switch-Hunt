"""游戏入口（兼容旧文件名）。"""
from switch_hunt.game.app import GameV8, main
from switch_hunt.game.player_v8 import PlayerV8
from switch_hunt.game.ghost_v8 import DQNGhostV8
from switch_hunt.game.treasure_v8 import TreasureV8
from switch_hunt.game.sound import SoundManager
from switch_hunt.enums import GhostState, LightState, GameState, LightMode

__all__ = [
    "GameV8", "PlayerV8", "DQNGhostV8", "TreasureV8", "SoundManager",
    "GhostState", "LightState", "GameState", "LightMode", "main",
]

if __name__ == "__main__":
    main()
