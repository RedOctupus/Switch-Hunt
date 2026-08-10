"""迷宫地图：DFS 生成、碰撞、渲染。"""
from __future__ import annotations

import random
from typing import List, Tuple, Optional

import pygame

from switch_hunt.constants import (
    MAP_WIDTH, MAP_HEIGHT, TILE_SIZE,
    COLOR_WALL, COLOR_WALL_BORDER, COLOR_FLOOR, COLOR_BLACK,
)
from switch_hunt.utils import clamp, distance

class Map:
    """
    迷宫地图类
    负责迷宫生成、碰撞检测和渲染
    """

    def __init__(self, width: int = MAP_WIDTH, height: int = MAP_HEIGHT):
        """
        初始化地图

        参数:
            width: 地图宽度（格子数）
            height: 地图高度（格子数）
        """
        self.width = width
        self.height = height
        # 创建二维数组，初始全部为墙壁（1=墙，0=空地）
        self.grid = [[1 for _ in range(width)] for _ in range(height)]
        # 生成迷宫
        self.generate_maze()

    def generate_maze(self) -> None:
        """
        使用深度优先回溯算法生成迷宫
        确保所有通道连通，单格宽度
        """
        # 从起点开始（必须是奇数坐标，确保在通道上）
        start_x, start_y = 1, 1
        self.grid[start_y][start_x] = 0  # 标记起点为空地

        # 使用栈来记录路径
        stack = [(start_x, start_y)]

        # 定义四个方向的移动（上、下、左、右），每次移动2格
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]

        # 深度优先搜索生成迷宫
        while stack:
            current_x, current_y = stack[-1]

            # 获取所有未访问的邻居
            neighbors = []
            for dx, dy in directions:
                next_x = current_x + dx
                next_y = current_y + dy

                # 检查是否在边界内且是墙壁（未访问）
                if (0 < next_x < self.width - 1 and 
                    0 < next_y < self.height - 1 and 
                    self.grid[next_y][next_x] == 1):
                    neighbors.append((next_x, next_y, dx, dy))

            if neighbors:
                # 随机选择一个邻居
                next_x, next_y, dx, dy = random.choice(neighbors)

                # 挖通当前位置到邻居之间的墙壁
                wall_x = current_x + dx // 2
                wall_y = current_y + dy // 2
                self.grid[wall_y][wall_x] = 0

                # 标记邻居为空地
                self.grid[next_y][next_x] = 0

                # 将邻居压入栈
                stack.append((next_x, next_y))
            else:
                # 没有未访问的邻居，回溯
                stack.pop()

    def is_wall(self, grid_x: int, grid_y: int) -> bool:
        """
        检查指定网格位置是否为墙壁

        参数:
            grid_x: 网格X坐标
            grid_y: 网格Y坐标

        返回:
            True如果是墙壁，False否则
        """
        # 检查边界
        if grid_x < 0 or grid_x >= self.width or grid_y < 0 or grid_y >= self.height:
            return True  # 边界外视为墙壁
        return self.grid[grid_y][grid_x] == 1

    def is_empty(self, grid_x: int, grid_y: int) -> bool:
        """
        检查指定网格位置是否为空地

        参数:
            grid_x: 网格X坐标
            grid_y: 网格Y坐标

        返回:
            True如果是空地，False否则
        """
        return not self.is_wall(grid_x, grid_y)

    def get_random_empty_position(self) -> Tuple[int, int]:
        """
        获取一个随机的空地位置

        返回:
            (grid_x, grid_y): 随机空地网格坐标
        """
        empty_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 0:
                    empty_positions.append((x, y))
        return random.choice(empty_positions)

    def get_wall_rects_around(self, pixel_x: float, pixel_y: float, radius: float) -> List[pygame.Rect]:
        """
        获取指定位置周围的所有墙壁矩形（用于碰撞检测优化）

        参数:
            pixel_x: 像素X坐标
            pixel_y: 像素Y坐标
            radius: 检测半径

        返回:
            墙壁矩形列表
        """
        wall_rects = []
        # 计算需要检测的网格范围
        min_grid_x = int((pixel_x - radius) // TILE_SIZE) - 1
        max_grid_x = int((pixel_x + radius) // TILE_SIZE) + 1
        min_grid_y = int((pixel_y - radius) // TILE_SIZE) - 1
        max_grid_y = int((pixel_y + radius) // TILE_SIZE) + 1

        # 限制在有效范围内
        min_grid_x = max(0, min_grid_x)
        max_grid_x = min(self.width - 1, max_grid_x)
        min_grid_y = max(0, min_grid_y)
        max_grid_y = min(self.height - 1, max_grid_y)

        # 收集范围内所有墙壁的矩形
        for gy in range(min_grid_y, max_grid_y + 1):
            for gx in range(min_grid_x, max_grid_x + 1):
                if self.grid[gy][gx] == 1:
                    wall_rect = pygame.Rect(
                        gx * TILE_SIZE, gy * TILE_SIZE,
                        TILE_SIZE, TILE_SIZE
                    )
                    wall_rects.append(wall_rect)

        return wall_rects

    def circle_rect_collision(self, circle_x: float, circle_y: float, 
                              radius: float, rect: pygame.Rect) -> Tuple[bool, float, float]:
        """
        检测圆形与矩形的碰撞

        参数:
            circle_x: 圆心X坐标
            circle_y: 圆心Y坐标
            radius: 圆半径
            rect: 矩形对象

        返回:
            (是否碰撞, 最近点X, 最近点Y)
        """
        # 找到矩形上距离圆心最近的点
        closest_x = clamp(circle_x, rect.left, rect.right)
        closest_y = clamp(circle_y, rect.top, rect.bottom)

        # 计算圆心到最近点的距离
        dist = distance(circle_x, circle_y, closest_x, closest_y)

        # 如果距离小于半径，则发生碰撞
        is_colliding = dist < radius

        return (is_colliding, closest_x, closest_y)

    def check_collision(self, pixel_x: float, pixel_y: float, radius: float) -> Tuple[bool, List]:
        """
        检查圆形碰撞箱与所有墙壁的碰撞

        参数:
            pixel_x: 圆心X坐标
            pixel_y: 圆心Y坐标
            radius: 圆半径

        返回:
            (是否碰撞, 碰撞信息列表)
        """
        collisions = []
        wall_rects = self.get_wall_rects_around(pixel_x, pixel_y, radius)

        for wall_rect in wall_rects:
            is_colliding, closest_x, closest_y = self.circle_rect_collision(
                pixel_x, pixel_y, radius, wall_rect
            )
            if is_colliding:
                collisions.append((wall_rect, closest_x, closest_y))

        return (len(collisions) > 0, collisions)

    def resolve_collision_slide(self, pixel_x: float, pixel_y: float, 
                                 radius: float, dx: float, dy: float) -> Tuple[float, float]:
        """
        滑移碰撞响应 - 让玩家沿着墙壁滑动

        参数:
            pixel_x: 当前圆心X坐标
            pixel_y: 当前圆心Y坐标
            radius: 圆半径
            dx: 尝试移动的X距离
            dy: 尝试移动的Y距离

        返回:
            (new_x, new_y): 滑动后的新位置
        """
        # 目标位置
        target_x = pixel_x + dx
        target_y = pixel_y + dy

        # 先尝试完整移动
        is_colliding, _ = self.check_collision(target_x, target_y, radius)

        if not is_colliding:
            return (target_x, target_y)

        # 有碰撞，尝试分别移动X和Y方向
        collide_x, _ = self.check_collision(pixel_x + dx, pixel_y, radius)
        collide_y, _ = self.check_collision(pixel_x, pixel_y + dy, radius)

        new_x, new_y = pixel_x, pixel_y

        if not collide_x:
            new_x = pixel_x + dx
        if not collide_y:
            new_y = pixel_y + dy

        return (new_x, new_y)

    def render(self, screen: pygame.Surface, camera_offset: Tuple[int, int] = (0, 0)):
        """
        渲染迷宫到屏幕

        参数:
            screen: Pygame屏幕对象
            camera_offset: 相机偏移量
        """
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(
                    x * TILE_SIZE + camera_offset[0],
                    y * TILE_SIZE + camera_offset[1],
                    TILE_SIZE, TILE_SIZE
                )
                if self.grid[y][x] == 1:
                    # 墙壁
                    pygame.draw.rect(screen, COLOR_WALL, rect)
                    pygame.draw.rect(screen, COLOR_WALL_BORDER, rect, 2)
                else:
                    # 空地
                    pygame.draw.rect(screen, COLOR_FLOOR, rect)


# =============================================================================
# 第五部分：A*寻路算法
# =============================================================================
