"""开关猎杀 (Switch Hunt) — 工程化包入口。"""

__version__ = "8.25"

from switch_hunt.enums import GameState, LightMode, LightState, GhostState
from switch_hunt.constants import TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT

__all__ = [
    "GameState",
    "LightMode",
    "LightState",
    "GhostState",
    "TILE_SIZE",
    "MAP_WIDTH",
    "MAP_HEIGHT",
    "SCREEN_WIDTH",
    "SCREEN_HEIGHT",
    "__version__",
]
