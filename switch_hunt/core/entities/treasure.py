"""宝藏实体。"""
from __future__ import annotations

from typing import Tuple

import pygame

from switch_hunt.constants import (
    TILE_SIZE, TREASURE_ENERGY_RESTORE, COLOR_GOLD, COLOR_YELLOW, COLOR_WHITE,
)
from switch_hunt.utils import grid_to_pixel, distance
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from switch_hunt.core.entities.player import Player

class Treasure:
    """
    宝藏类
    负责宝藏的生成、渲染和拾取检测
    """

    def __init__(self, grid_x: int, grid_y: int):
        """
        初始化宝藏

        参数:
            grid_x: 格子X坐标
            grid_y: 格子Y坐标
        """
        self.grid_pos = (grid_x, grid_y)
        self.pixel_pos = grid_to_pixel(grid_x, grid_y)
        self.radius = TILE_SIZE // 3
        self.collected = False
        self.animation_offset = 0.0
        self.animation_speed = 3.0

    def update(self, dt: float):
        """
        更新宝藏动画

        参数:
            dt: 时间增量（秒）
        """
        self.animation_offset += self.animation_speed * dt

    def check_pickup(self, player: Player) -> bool:
        """
        检查玩家是否可以拾取宝藏

        参数:
            player: 玩家对象

        返回:
            是否成功拾取
        """
        if self.collected:
            return False

        player_pos = player.get_pixel_pos()
        dist = distance(self.pixel_pos[0], self.pixel_pos[1], 
                       player_pos[0], player_pos[1])

        if dist < (player.radius + self.radius):
            self.collected = True
            return True

        return False

    def render(self, screen: pygame.Surface, camera_offset: Tuple[int, int] = (0, 0)):
        """
        渲染宝藏

        参数:
            screen: Pygame屏幕表面
            camera_offset: 相机偏移量
        """
        if self.collected:
            return

        # 计算动画偏移
        float_offset = math.sin(self.animation_offset) * 3

        screen_x = int(self.pixel_pos[0] + camera_offset[0])
        screen_y = int(self.pixel_pos[1] + camera_offset[1] + float_offset)

        # 绘制宝藏（菱形）
        points = [
            (screen_x, screen_y - self.radius),
            (screen_x + self.radius, screen_y),
            (screen_x, screen_y + self.radius),
            (screen_x - self.radius, screen_y),
        ]
        pygame.draw.polygon(screen, COLOR_GOLD, points)
        pygame.draw.polygon(screen, COLOR_YELLOW, points, 2)

        # 绘制闪光效果
        sparkle_size = 3 + int(math.sin(self.animation_offset * 2) * 2)
        pygame.draw.circle(screen, COLOR_WHITE,
                          (screen_x - self.radius // 2, screen_y - self.radius // 2),
                          sparkle_size)


# =============================================================================
# 第八部分：鬼AI类
# =============================================================================
