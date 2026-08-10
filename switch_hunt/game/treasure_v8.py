"""V8 宝藏（拾取重置光源次数）。"""
from switch_hunt.core.entities.treasure import Treasure


class TreasureV8(Treasure):
    """V8宝藏"""
    def __init__(self, grid_x, grid_y):
        super().__init__(grid_x, grid_y)

