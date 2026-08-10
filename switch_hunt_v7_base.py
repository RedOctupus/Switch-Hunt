"""兼容层：旧基础库导入。"""
from switch_hunt.constants import *  # noqa: F401,F403
from switch_hunt.enums import GameState, LightMode, LightState, GhostState
from switch_hunt.utils import grid_to_pixel, pixel_to_grid, distance, clamp
from switch_hunt.core.map import Map
from switch_hunt.core.pathfinding import AStarPathfinder
from switch_hunt.core.entities.player import Player
from switch_hunt.core.entities.ghost import Ghost
from switch_hunt.core.entities.treasure import Treasure
from switch_hunt.core.visibility import VisibilitySystem
from switch_hunt.game.ui_base import UISystem
from switch_hunt.game.manager import GameManager

__all__ = [
    "GameState", "LightMode", "LightState", "GhostState",
    "grid_to_pixel", "pixel_to_grid", "distance", "clamp",
    "Map", "AStarPathfinder", "Player", "Ghost", "Treasure",
    "VisibilitySystem", "UISystem", "GameManager",
]
