"""迷雾 / 视野系统。"""
from __future__ import annotations

import math
from typing import Set, Tuple, TYPE_CHECKING

import pygame

from switch_hunt.constants import TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, COLOR_BLACK
from switch_hunt.utils import pixel_to_grid

if TYPE_CHECKING:
    from switch_hunt.core.map import Map

class VisibilitySystem:
    """
    可见性系统
    负责光照计算、迷雾渲染和墙壁可见度设置
    """

    def __init__(self, game_map: Map, screen_width: int, screen_height: int):
        """
        初始化可见性系统

        参数:
            game_map: 游戏地图对象
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
        """
        self.game_map = game_map
        self.screen_width = screen_width
        self.screen_height = screen_height

        # 迷雾表面
        self.fog_surface = pygame.Surface((screen_width, screen_height))
        self.fog_surface.fill(COLOR_BLACK)

        # 光照表面
        self.light_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)

        # 墙壁可见度
        self.wall_visibility = 0.4

        # 已探索区域
        self.explored = [[False for _ in range(game_map.width)]
                        for _ in range(game_map.height)]

    def update(self, player: Player):
        """
        更新可见性系统

        参数:
            player: 玩家对象
        """
        player_grid = player.get_grid_pos()
        light_radius = player.light_radius

        # 标记玩家周围区域为已探索
        for dy in range(-light_radius, light_radius + 1):
            for dx in range(-light_radius, light_radius + 1):
                gx = player_grid[0] + dx
                gy = player_grid[1] + dy

                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= light_radius:
                    if 0 <= gx < self.game_map.width and 0 <= gy < self.game_map.height:
                        if self._has_line_of_sight(player_grid, (gx, gy)):
                            self.explored[gy][gx] = True

    def _has_line_of_sight(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """
        检查两点之间是否有视线（Bresenham算法）

        参数:
            start: 起点格子坐标
            end: 终点格子坐标

        返回:
            是否有视线
        """
        x0, y0 = start
        x1, y1 = end

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if self.game_map.is_wall(x0, y0) and (x0, y0) != start:
                return False

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return True

    def render(self, screen: pygame.Surface, player: Player,
               camera_offset: Tuple[int, int] = (0, 0)):
        """
        渲染迷雾和光照
        方案：每个格子根据状态绘制不同透明度的黑色覆盖
        
        参数:
            screen: Pygame屏幕表面
            player: 玩家对象
            camera_offset: 相机偏移量
        """
        # 创建迷雾表面
        fog_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        fog_surface.fill((0, 0, 0, 0))  # 全透明
        
        player_pos = player.get_pixel_pos()
        light_radius_px = player.light_radius * TILE_SIZE
        center_x = int(player_pos[0] + camera_offset[0])
        center_y = int(player_pos[1] + camera_offset[1])
        player_grid = player.get_grid_pos()

        # 遍历每个格子确定其迷雾浓度
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                rect = pygame.Rect(
                    x * TILE_SIZE + camera_offset[0],
                    y * TILE_SIZE + camera_offset[1],
                    TILE_SIZE, TILE_SIZE
                )
                
                # 计算格子中心到玩家的距离
                grid_pixel_x = x * TILE_SIZE + TILE_SIZE // 2 + camera_offset[0]
                grid_pixel_y = y * TILE_SIZE + TILE_SIZE // 2 + camera_offset[1]
                dist = math.sqrt((grid_pixel_x - center_x)**2 + (grid_pixel_y - center_y)**2)
                
                # 检查是否在光照范围内且有视线
                in_light = False
                if dist <= light_radius_px:
                    if self._has_line_of_sight(player_grid, (x, y)):
                        in_light = True
                
                if in_light:
                    # 光照范围内：根据距离计算暗度（中心亮，边缘暗）
                    darkness = int(180 * (dist / light_radius_px))
                    darkness = max(0, min(80, darkness))  # 最多80/255的暗度
                    pygame.draw.rect(fog_surface, (0, 0, 0, darkness), rect)
                elif self.explored[y][x]:
                    # 已探索但无光：中等暗度
                    pygame.draw.rect(fog_surface, (0, 0, 0, 160), rect)
                else:
                    # 未探索：完全遮蔽
                    pygame.draw.rect(fog_surface, (0, 0, 0, 235), rect)

        # 将迷雾应用到屏幕
        screen.blit(fog_surface, (0, 0))


# =============================================================================
# 第十部分：DQN接口（为强化学习预留）
# =============================================================================
