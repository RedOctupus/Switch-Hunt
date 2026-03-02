#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《开关猎杀》游戏 - 完整版
========================
游戏类型：11×11迷宫捉迷藏游戏
技术栈：Pygame + Python 3.x

游戏特色：
- 随机生成迷宫（深度优先回溯算法）
- 玩家携带光源探索迷雾
- 鬼AI使用A*寻路追踪玩家
- 强化光源可定身鬼
- 收集4个宝藏获胜
- 为DQN强化学习预留接口

操作说明：
- WASD/方向键：移动
- 空格键：开启/关闭强化光源
- P键：暂停游戏
- ESC键：返回菜单
- F1键：切换作弊模式

作者：AI游戏开发团队
版本：1.0.0
"""

import pygame
import numpy as np
import heapq
import random
import math
import os
from typing import List, Tuple, Dict, Optional
from enum import Enum, auto

# =============================================================================
# 第一部分：常量定义
# =============================================================================

# 屏幕设置
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

# 颜色定义
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (128, 128, 128)
COLOR_DARK_GRAY = (64, 64, 64)
COLOR_LIGHT_GRAY = (100, 100, 100)
COLOR_YELLOW = (255, 255, 0)
COLOR_BLUE = (0, 100, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_ORANGE = (255, 165, 0)
COLOR_GOLD = (255, 215, 0)
COLOR_CYAN = (0, 255, 255)

# 墙壁颜色
COLOR_WALL = (60, 60, 80)      # 墙壁填充色
COLOR_WALL_BORDER = (100, 100, 120)  # 墙壁边框色
COLOR_FLOOR = (80, 80, 100)    # 地板颜色（调亮以便在迷雾中可见）

# 鬼颜色
COLOR_GHOST_NORMAL = (255, 100, 100)   # 鬼正常状态颜色
COLOR_GHOST_STUNNED = (100, 100, 255)  # 鬼定身状态颜色


# =============================================================================
# 第二部分：枚举类型
# =============================================================================

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


# V7: 光源系统配置
LIGHT_V7 = {
    'max_charges': 5,           # 最大使用次数
    'initial_charges': 5,       # 初始次数
    'active_duration': 5.0,     # 激活持续时间(秒)
    'cooldown_duration': 3.0,   # 冷却时间(秒)
    'radius': 3,                # 照亮半径(格子数)
}


class GhostState(Enum):
    """鬼状态枚举"""
    NORMAL = "normal"   # 正常状态
    STUNNED = "stunned" # 定身状态


# =============================================================================
# 第三部分：辅助函数
# =============================================================================

def grid_to_pixel(grid_x: int, grid_y: int) -> Tuple[int, int]:
    """
    将网格坐标转换为像素坐标（格子中心点）

    参数:
        grid_x: 网格X坐标
        grid_y: 网格Y坐标

    返回:
        (pixel_x, pixel_y): 像素坐标（格子中心）
    """
    pixel_x = grid_x * TILE_SIZE + TILE_SIZE // 2
    pixel_y = grid_y * TILE_SIZE + TILE_SIZE // 2
    return (pixel_x, pixel_y)


def pixel_to_grid(pixel_x: float, pixel_y: float) -> Tuple[int, int]:
    """
    将像素坐标转换为网格坐标

    参数:
        pixel_x: 像素X坐标
        pixel_y: 像素Y坐标

    返回:
        (grid_x, grid_y): 网格坐标
    """
    grid_x = int(pixel_x // TILE_SIZE)
    grid_y = int(pixel_y // TILE_SIZE)
    grid_x = max(0, min(grid_x, MAP_WIDTH - 1))
    grid_y = max(0, min(grid_y, MAP_HEIGHT - 1))
    return (grid_x, grid_y)


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    计算两点之间的欧几里得距离

    参数:
        x1, y1: 第一个点的坐标
        x2, y2: 第二个点的坐标

    返回:
        两点之间的距离
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    将值限制在指定范围内

    参数:
        value: 要限制的值
        min_val: 最小值
        max_val: 最大值

    返回:
        限制后的值
    """
    return max(min_val, min(value, max_val))


# =============================================================================
# 第四部分：地图类（迷宫生成与碰撞系统）
# =============================================================================

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

class AStarPathfinder:
    """A*寻路算法实现类"""

    def __init__(self, map_obj: Map):
        """
        初始化A*寻路器

        参数:
            map_obj: 地图对象
        """
        self.map = map_obj

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        启发函数：使用曼哈顿距离

        参数:
            a: 起点格子坐标
            b: 终点格子坐标

        返回:
            估计距离
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A*寻路主函数

        参数:
            start: 起点格子坐标
            goal: 目标格子坐标

        返回:
            路径列表，每个元素是格子坐标
        """
        # 如果起点或终点是墙壁，返回空路径
        if self.map.is_wall(start[0], start[1]) or self.map.is_wall(goal[0], goal[1]):
            return []

        if start == goal:
            return [start]

        # 初始化开放列表（优先队列）
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start))

        # 记录每个节点的来源
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # g_score: 从起点到当前节点的实际代价
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        # f_score: 估计的总代价
        f_score: Dict[Tuple[int, int], int] = {start: self.heuristic(start, goal)}

        # 开放列表中的节点集合
        open_set_hash = {start}

        # 四个移动方向
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while open_set:
            # 取出f_score最小的节点
            current_f, _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            # 到达目标，重建路径
            if current == goal:
                return self._reconstruct_path(came_from, current)

            # 遍历邻居
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                # 检查邻居是否可行走
                if not self.map.is_wall(neighbor[0], neighbor[1]):
                    tentative_g = g_score[current] + 1

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)

                        if neighbor not in open_set_hash:
                            counter += 1
                            heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                            open_set_hash.add(neighbor)

        # 没有找到路径
        return []

    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        重建路径

        参数:
            came_from: 记录每个节点来源的字典
            current: 终点节点

        返回:
            从起点到终点的路径列表
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# =============================================================================
# 第六部分：玩家类
# =============================================================================

class Player:
    """
    玩家类
    负责玩家的移动、碰撞检测、光源系统和能量管理
    """

    def __init__(self, x: float, y: float, game_map: Map):
        """
        初始化玩家

        参数:
            x: 初始X坐标（像素）
            y: 初始Y坐标（像素）
            game_map: 游戏地图对象
        """
        # 位置和移动
        self.pos = [x, y]  # 玩家位置 [x, y]
        self.velocity = [0.0, 0.0]  # 速度向量
        self.radius = PLAYER_RADIUS  # 碰撞半径
        self.speed = PLAYER_SPEED  # 移动速度
        self.game_map = game_map  # 地图引用

        # 光源系统
        self.light_mode = LightMode.NORMAL
        self.light_radius = LIGHT_RADIUS_NORMAL

        # 能量系统
        self.energy = PLAYER_MAX_ENERGY
        self.max_energy = PLAYER_MAX_ENERGY
        self.energy_decay_rate = PLAYER_ENERGY_DECAY
        self.energy_regen_rate = PLAYER_ENERGY_REGEN

        # 输入状态
        self.keys_pressed = {
            'up': False, 'down': False,
            'left': False, 'right': False,
            'enhance': False
        }

    def handle_input(self, event: pygame.event.Event):
        """
        处理输入事件

        参数:
            event: Pygame事件对象
        """
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.keys_pressed['up'] = True
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.keys_pressed['down'] = True
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                self.keys_pressed['left'] = True
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                self.keys_pressed['right'] = True
            elif event.key == pygame.K_SPACE:
                self.keys_pressed['enhance'] = True

        elif event.type == pygame.KEYUP:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.keys_pressed['up'] = False
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.keys_pressed['down'] = False
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                self.keys_pressed['left'] = False
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                self.keys_pressed['right'] = False
            elif event.key == pygame.K_SPACE:
                self.keys_pressed['enhance'] = False

    def update(self, dt: float):
        """
        更新玩家状态（平滑移动）

        参数:
            dt: 时间增量（秒）
        """
        # 计算移动方向
        dx = 0.0
        dy = 0.0

        if self.keys_pressed['up']:
            dy -= 1.0
        if self.keys_pressed['down']:
            dy += 1.0
        if self.keys_pressed['left']:
            dx -= 1.0
        if self.keys_pressed['right']:
            dx += 1.0

        # 归一化移动向量（防止斜向移动过快）
        if dx != 0 or dy != 0:
            length = math.sqrt(dx * dx + dy * dy)
            dx /= length
            dy /= length

        # 计算目标位置
        target_x = self.pos[0] + dx * self.speed * dt
        target_y = self.pos[1] + dy * self.speed * dt

        # X轴碰撞检测
        if self._can_move_to(target_x, self.pos[1]):
            self.pos[0] = target_x

        # Y轴碰撞检测
        if self._can_move_to(self.pos[0], target_y):
            self.pos[1] = target_y

        # 更新光源模式
        self._update_light_mode(dt)

    def _can_move_to(self, x: float, y: float) -> bool:
        """
        检查是否可以移动到指定位置

        参数:
            x: 目标X坐标
            y: 目标Y坐标

        返回:
            是否可以移动
        """
        # 使用地图的碰撞检测
        is_colliding, _ = self.game_map.check_collision(x, y, self.radius)
        return not is_colliding

    def _update_light_mode(self, dt: float):
        """
        更新光源模式和能量

        参数:
            dt: 时间增量（秒）
        """
        # 检查是否要开启强化光源
        if self.keys_pressed['enhance'] and self.energy > 0:
            self.light_mode = LightMode.ENHANCED
            self.light_radius = LIGHT_RADIUS_ENHANCED
            # 消耗能量
            self.energy -= self.energy_decay_rate * dt
            self.energy = max(0, self.energy)
        else:
            # 普通模式
            self.light_mode = LightMode.NORMAL
            self.light_radius = LIGHT_RADIUS_NORMAL
            # 修改：取消能量自动恢复，整局游戏光源只可被消耗
            # if self.energy < self.max_energy:
            #     self.energy += self.energy_regen_rate * dt
            #     self.energy = min(self.max_energy, self.energy)

    def get_grid_pos(self) -> Tuple[int, int]:
        """
        获取玩家所在的格子坐标

        返回:
            格子坐标 (grid_x, grid_y)
        """
        return pixel_to_grid(self.pos[0], self.pos[1])

    def get_pixel_pos(self) -> Tuple[float, float]:
        """
        获取玩家的像素坐标

        返回:
            像素坐标 (x, y)
        """
        return (self.pos[0], self.pos[1])

    def is_enhanced_light(self) -> bool:
        """
        检查是否处于强化光源模式

        返回:
            是否为强化光源
        """
        return self.light_mode == LightMode.ENHANCED

    def add_energy(self, amount: float):
        """
        增加能量

        参数:
            amount: 增加的能量值
        """
        self.energy = min(self.max_energy, self.energy + amount)

    def render(self, screen: pygame.Surface, camera_offset: Tuple[int, int] = (0, 0)):
        """
        渲染玩家

        参数:
            screen: Pygame屏幕表面
            camera_offset: 相机偏移量
        """
        screen_x = int(self.pos[0] + camera_offset[0])
        screen_y = int(self.pos[1] + camera_offset[1])

        # 绘制玩家圆形
        pygame.draw.circle(screen, COLOR_BLUE, (screen_x, screen_y), self.radius)

        # 绘制光源指示器（外圈）
        light_color = COLOR_ORANGE if self.is_enhanced_light() else COLOR_YELLOW
        light_radius_px = self.light_radius * TILE_SIZE
        pygame.draw.circle(screen, light_color, (screen_x, screen_y), light_radius_px, 2)


# =============================================================================
# 第七部分：宝藏类
# =============================================================================

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

class Ghost:
    """
    鬼AI类
    包含寻路、移动、状态机、碰撞检测
    """

    def __init__(self, x: float, y: float, player_speed: float, game_map: Map):
        """
        初始化鬼

        参数:
            x: 初始x坐标（像素）
            y: 初始y坐标（像素）
            player_speed: 玩家速度（用于计算鬼速度）
            game_map: 地图对象
        """
        # 位置属性
        self.pos = [float(x), float(y)]
        self.grid_pos = pixel_to_grid(x, y)

        # 速度属性
        self.player_speed = player_speed
        self.speed = player_speed * GHOST_SPEED_RATIO

        # 状态属性
        self.state = GhostState.NORMAL
        self.stun_timer = 0.0

        # 寻路属性
        self.pathfinder = AStarPathfinder(game_map)
        self.path: List[Tuple[int, int]] = []
        self.path_update_timer = 0.0
        self.path_update_interval = 0.5  # 每0.5秒更新一次路径

        # 碰撞属性
        self.radius = GHOST_RADIUS
        self.game_map = game_map

        # DQN相关
        self.last_action = None
        self.step_count = 0

    def update(self, dt: float, player: Player):
        """
        更新鬼的状态和位置

        参数:
            dt: 时间增量（秒）
            player: 玩家对象
        """
        self.step_count += 1
        self.grid_pos = pixel_to_grid(self.pos[0], self.pos[1])

        # 检查定身状态是否结束
        self._update_stun_state(dt)

        # 只有在正常状态才移动
        if self.state == GhostState.NORMAL:
            self._update_movement(dt, player)
    
    def _update_stun_state(self, dt: float):
        """更新定身状态（不处理移动）"""
        if self.state == GhostState.STUNNED:
            self.stun_timer -= dt
            if self.stun_timer <= 0:
                self.state = GhostState.NORMAL
                self.stun_timer = 0
    
    def update_for_dqn_training(self, dt: float, player: Player):
        """
        DQN训练专用更新（只更新状态，不移动）
        
        在DQN训练中，移动由apply_action控制，不应使用A*寻路覆盖。
        此方法只更新：
        - step_count
        - grid_pos
        - 定身状态计时器
        
        参数:
            dt: 时间增量（秒）
            player: 玩家对象（预留参数，实际不使用）
        """
        self.step_count += 1
        self.grid_pos = pixel_to_grid(self.pos[0], self.pos[1])
        self._update_stun_state(dt)

    def _update_movement(self, dt: float, player: Player):
        """
        更新移动（平滑沿路径移动）

        参数:
            dt: 时间增量（秒）
            player: 玩家对象
        """
        # 定期更新路径
        self.path_update_timer += dt
        if self.path_update_timer >= self.path_update_interval:
            self.find_path(player.get_grid_pos())
            self.path_update_timer = 0.0

        # 如果有路径，沿路径移动
        if self.path and len(self.path) > 1:
            target_grid = self.path[1]  # 下一个格子
            target_x, target_y = grid_to_pixel(target_grid[0], target_grid[1])

            # 计算到目标的方向向量
            dx = target_x - self.pos[0]
            dy = target_y - self.pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            # 如果到达目标点，移动到下一个路径点
            if dist < 5:
                self.path.pop(0)
                return

            # 归一化方向向量并移动
            if dist > 0:
                dx /= dist
                dy /= dist

            move_distance = self.speed * dt
            new_x = self.pos[0] + dx * move_distance
            new_y = self.pos[1] + dy * move_distance

            # 检查墙壁碰撞
            if not self._check_wall_collision(new_x, new_y):
                self.pos[0] = new_x
                self.pos[1] = new_y

    def _check_wall_collision(self, x: float, y: float) -> bool:
        """
        检查指定位置是否与墙壁碰撞

        参数:
            x: x坐标
            y: y坐标

        返回:
            是否碰撞
        """
        offsets = [(-self.radius, -self.radius), (self.radius, -self.radius),
                   (-self.radius, self.radius), (self.radius, self.radius)]

        for dx, dy in offsets:
            grid_x = int((x + dx) // TILE_SIZE)
            grid_y = int((y + dy) // TILE_SIZE)
            if self.game_map.is_wall(grid_x, grid_y):
                return True
        return False

    def find_path(self, target_grid: Tuple[int, int]):
        """
        使用A*算法寻找路径

        参数:
            target_grid: 目标格子坐标
        """
        self.path = self.pathfinder.find_path(self.grid_pos, target_grid)

    def freeze(self, duration: float):
        """
        定身鬼

        参数:
            duration: 定身持续时间（秒）
        """
        self.state = GhostState.STUNNED
        self.stun_timer = duration

    def check_collision(self, player: Player) -> bool:
        """
        检测与玩家的碰撞

        参数:
            player: 玩家对象

        返回:
            是否碰撞
        """
        player_pos = player.get_pixel_pos()
        dist = distance(self.pos[0], self.pos[1], player_pos[0], player_pos[1])
        return dist < (self.radius + player.radius)

    def apply_action(self, action: int, dt: float):
        """
        根据DQN动作移动鬼（用于训练）

        参数:
            action: 动作索引（0:上, 1:下, 2:左, 3:右）
            dt: 时间增量（秒）
        """
        if self.state == GhostState.STUNNED:
            return

        self.last_action = action
        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = actions[action]

        move_distance = self.speed * dt
        new_x = self.pos[0] + dx * move_distance
        new_y = self.pos[1] + dy * move_distance

        if not self._check_wall_collision(new_x, new_y):
            self.pos[0] = new_x
            self.pos[1] = new_y

    def render(self, screen: pygame.Surface, camera_offset: Tuple[int, int] = (0, 0)):
        """
        渲染鬼

        参数:
            screen: Pygame屏幕表面
            camera_offset: 相机偏移量
        """
        # 根据状态选择颜色
        color = COLOR_GHOST_STUNNED if self.state == GhostState.STUNNED else COLOR_GHOST_NORMAL

        screen_x = int(self.pos[0] + camera_offset[0])
        screen_y = int(self.pos[1] + camera_offset[1])

        # 绘制鬼（圆形）
        pygame.draw.circle(screen, color, (screen_x, screen_y), self.radius)

        # 绘制碰撞箱轮廓
        pygame.draw.circle(screen, COLOR_WHITE, (screen_x, screen_y), self.radius, 2)

        # 绘制状态指示
        if self.state == GhostState.STUNNED:
            # 定身时绘制闪电符号
            pygame.draw.line(screen, COLOR_YELLOW,
                            (screen_x - 5, screen_y - 8),
                            (screen_x + 5, screen_y), 2)
            pygame.draw.line(screen, COLOR_YELLOW,
                            (screen_x + 5, screen_y),
                            (screen_x - 5, screen_y + 8), 2)


# =============================================================================
# 第九部分：可见性/迷雾系统
# =============================================================================

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

class DQNInterface:
    """DQN接口类 - 提供状态编码、动作空间和奖励函数"""

    @staticmethod
    def get_state_encoding(game_map: Map, player: Player, ghost: Ghost) -> np.ndarray:
        """
        v6.5: 获取状态编码（6通道21×21矩阵）
        改为全局视野（以地图左上角为中心），而非以鬼为中心

        参数:
            game_map: 地图对象
            player: 玩家对象
            ghost: 鬼对象

        返回:
            6×21×21的numpy数组（覆盖整个地图）
        """
        state = np.zeros((STATE_CHANNELS, STATE_SIZE, STATE_SIZE), dtype=np.float32)

        # v6.5: 改为全局视野，直接遍历整个地图（0,0）到（20,20）
        # 不再以鬼为中心，而是以地图左上角（0,0）为基准
        
        # v6.5: 计算A*最短路径（第6维通道）
        player_pos = player.get_grid_pos()
        path = ghost.pathfinder.find_path(ghost.grid_pos, player_pos)
        path_set = set(path) if path else set()

        for grid_y in range(game_map.height):
            for grid_x in range(game_map.width):
                state_x = grid_x
                state_y = grid_y

                # 通道0: 墙壁地图
                if game_map.is_wall(grid_x, grid_y):
                    state[0, state_y, state_x] = 1.0

                # 通道1: 鬼位置
                if (grid_x, grid_y) == ghost.grid_pos:
                    state[1, state_y, state_x] = 1.0

                # 通道2: 玩家位置
                if (grid_x, grid_y) == player.get_grid_pos():
                    state[2, state_y, state_x] = 1.0

                # 通道3: 强化光源覆盖区域
                if player.is_enhanced_light():
                    dist = math.sqrt((grid_x - player.get_grid_pos()[0])**2 +
                                    (grid_y - player.get_grid_pos()[1])**2)
                    if dist <= player.light_radius:
                        state[3, state_y, state_x] = 1.0 - (dist / player.light_radius)

                # 通道4: 强化光源开启标志
                if player.is_enhanced_light():
                    state[4, state_y, state_x] = 1.0

                # 通道5: A*最短路径（v6.5保持）
                # 路径上的格子标记为1.0，提供密集的学习信号
                if (grid_x, grid_y) in path_set:
                    state[5, state_y, state_x] = 1.0

        return state

    @staticmethod
    def get_action_space() -> List[int]:
        """
        获取动作空间

        返回:
            动作列表 [0:上, 1:下, 2:左, 3:右]
        """
        return list(range(ACTION_SPACE))

    @staticmethod
    def calculate_reward(ghost: Ghost, player: Player, touched_player: bool = False,
                         was_stunned: bool = False) -> float:
        """
        计算奖励函数

        参数:
            ghost: 鬼对象
            player: 玩家对象
            touched_player: 是否触碰到玩家
            was_stunned: 是否被定身

        返回:
            奖励值
        """
        reward = 0.0

        # 触碰到玩家：+10
        if touched_player:
            reward += 10.0

        # 每步惩罚：-0.01
        reward -= 0.01

        # 被定身惩罚：-1
        if was_stunned:
            reward -= 1.0

        return reward


# =============================================================================
# 第十一部分：UI系统
# =============================================================================

class UISystem:
    """
    UI系统
    负责HUD、菜单和作弊模式界面
    """

    def __init__(self, screen_width: int, screen_height: int):
        """
        初始化UI系统

        参数:
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

        # 字体 - 尝试加载系统中文字体，失败则使用默认字体
        self.font_large = self._load_font(72)
        self.font_medium = self._load_font(48)
        self.font_small = self._load_font(36)
        
        # 如果系统没有中文字体，使用英文文本作为备选
        self.use_english = self.font_large is None or not self._check_font_support()
        if self.use_english:
            self.font_large = pygame.font.Font(None, 72)
            self.font_medium = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 36)
        
        # 菜单选项（根据字体支持决定语言）
        self.ai_mode = False  # AI演示模式
        self.difficulty = 'normal'  # 难度: easy, normal, hard
        
        if self.use_english:
            self.menu_options = [
                "Start Game", 
                "Difficulty: Normal",
                "AI Demo: OFF",
                "Cheat Mode: OFF", 
                "Quit"
            ]
        else:
            self.menu_options = [
                "开始游戏", 
                "难度: 普通",
                "AI演示: 关",
                "作弊模式: 关", 
                "退出"
            ]
        self.selected_option = 0

        # 作弊模式
        self.cheat_mode = False

        # HUD位置
        self.hud_margin = 20
        self.energy_bar_width = 200
        self.energy_bar_height = 20
    
    def _load_font(self, size: int) -> Optional[pygame.font.Font]:
        """尝试加载支持中文的字体"""
        # 常见的Windows中文字体路径
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑粗体
        ]
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    return pygame.font.Font(font_path, size)
            except:
                continue
        return None
    
    def _check_font_support(self) -> bool:
        """检查字体是否支持中文字符"""
        try:
            test_surface = self.font_large.render("测试", True, COLOR_WHITE)
            return test_surface.get_width() > 20
        except:
            return False

    def handle_menu_input(self, event: pygame.event.Event) -> Optional[str]:
        """
        处理菜单输入

        参数:
            event: Pygame事件对象

        返回:
            选择的操作，或None
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_option = (self.selected_option - 1) % len(self.menu_options)
            elif event.key == pygame.K_DOWN:
                self.selected_option = (self.selected_option + 1) % len(self.menu_options)
            elif event.key == pygame.K_RETURN:
                if self.selected_option == 0:
                    return "start"
                elif self.selected_option == 1:
                    # 切换难度: normal -> easy -> hard -> normal
                    if self.difficulty == 'normal':
                        self.difficulty = 'easy'
                    elif self.difficulty == 'easy':
                        self.difficulty = 'hard'
                    else:
                        self.difficulty = 'normal'
                    
                    if self.use_english:
                        diff_text = self.difficulty.capitalize()
                        self.menu_options[1] = f"Difficulty: {diff_text}"
                    else:
                        diff_map = {'easy': '简单', 'normal': '普通', 'hard': '困难'}
                        self.menu_options[1] = f"难度: {diff_map[self.difficulty]}"
                        
                elif self.selected_option == 2:
                    # 切换AI演示模式
                    self.ai_mode = not self.ai_mode
                    if self.use_english:
                        self.menu_options[2] = f"AI Demo: {'ON' if self.ai_mode else 'OFF'}"
                    else:
                        self.menu_options[2] = f"AI演示: {'开' if self.ai_mode else '关'}"
                        
                elif self.selected_option == 3:
                    self.cheat_mode = not self.cheat_mode
                    if self.use_english:
                        self.menu_options[3] = f"Cheat Mode: {'ON' if self.cheat_mode else 'OFF'}"
                    else:
                        self.menu_options[3] = f"作弊模式: {'开' if self.cheat_mode else '关'}"
                        
                elif self.selected_option == 4:
                    return "quit"
        return None

    def render_menu(self, screen: pygame.Surface):
        """
        渲染主菜单

        参数:
            screen: Pygame屏幕表面
        """
        screen.fill(COLOR_BLACK)

        # 标题
        if self.use_english:
            title = self.font_large.render("Switch Hunt", True, COLOR_GOLD)
            subtitle_text = ""
        else:
            title = self.font_large.render("开关猎杀", True, COLOR_GOLD)
            subtitle_text = "Switch Hunt"
        title_rect = title.get_rect(center=(self.screen_width // 2, 150))
        screen.blit(title, title_rect)

        # 副标题
        if subtitle_text:
            subtitle = self.font_small.render(subtitle_text, True, COLOR_GRAY)
            subtitle_rect = subtitle.get_rect(center=(self.screen_width // 2, 210))
            screen.blit(subtitle, subtitle_rect)

        # 菜单选项
        for i, option in enumerate(self.menu_options):
            color = COLOR_YELLOW if i == self.selected_option else COLOR_WHITE
            text = self.font_medium.render(option, True, color)
            rect = text.get_rect(center=(self.screen_width // 2, 350 + i * 60))
            screen.blit(text, rect)

        # 操作提示
        if self.use_english:
            hint_text = "Use Arrow Keys to select, Enter to confirm"
        else:
            hint_text = "使用方向键选择，回车确认"
        hint = self.font_small.render(hint_text, True, COLOR_GRAY)
        hint_rect = hint.get_rect(center=(self.screen_width // 2, self.screen_height - 100))
        screen.blit(hint, hint_rect)

    def render_hud(self, screen: pygame.Surface, player: Player,
                   treasures_collected: int, total_treasures: int):
        """
        渲染HUD

        参数:
            screen: Pygame屏幕表面
            player: 玩家对象
            treasures_collected: 已收集宝藏数
            total_treasures: 总宝藏数
        """
        # 宝藏计数
        if self.use_english:
            treasure_label = "Treasures"
        else:
            treasure_label = "宝藏"
        treasure_text = self.font_small.render(
            f"{treasure_label}: {treasures_collected}/{total_treasures}", True, COLOR_GOLD)
        screen.blit(treasure_text, (self.hud_margin, self.hud_margin))

        # 能量条背景
        bar_x = self.hud_margin
        bar_y = self.hud_margin + 40
        pygame.draw.rect(screen, COLOR_DARK_GRAY,
                        (bar_x, bar_y, self.energy_bar_width, self.energy_bar_height))

        # 能量条填充
        energy_ratio = player.energy / player.max_energy
        energy_width = int(self.energy_bar_width * energy_ratio)

        if energy_ratio > 0.5:
            energy_color = COLOR_GREEN
        elif energy_ratio > 0.25:
            energy_color = COLOR_YELLOW
        else:
            energy_color = COLOR_RED

        pygame.draw.rect(screen, energy_color,
                        (bar_x, bar_y, energy_width, self.energy_bar_height))

        # 能量条边框
        pygame.draw.rect(screen, COLOR_WHITE,
                        (bar_x, bar_y, self.energy_bar_width, self.energy_bar_height), 2)

        # 能量数值
        if self.use_english:
            energy_label = "Energy"
        else:
            energy_label = "能量"
        energy_text = self.font_small.render(
            f"{energy_label}: {int(player.energy)}/{player.max_energy}", True, COLOR_WHITE)
        screen.blit(energy_text, (bar_x + self.energy_bar_width + 10, bar_y - 2))

        # 光源模式指示
        if self.use_english:
            light_text = "Enhanced Light" if player.is_enhanced_light() else "Normal Light"
        else:
            light_text = "强化光源" if player.is_enhanced_light() else "普通光源"
        light_color = COLOR_ORANGE if player.is_enhanced_light() else COLOR_YELLOW
        light_surface = self.font_small.render(light_text, True, light_color)
        screen.blit(light_surface, (bar_x, bar_y + 30))

        # 作弊模式指示
        if self.cheat_mode:
            if self.use_english:
                cheat_label = "[Cheat Mode ON]"
            else:
                cheat_label = "[作弊模式开启]"
            cheat_text = self.font_small.render(cheat_label, True, COLOR_RED)
            screen.blit(cheat_text, (self.screen_width - 200, self.hud_margin))

    def render_pause(self, screen: pygame.Surface):
        """
        渲染暂停界面

        参数:
            screen: Pygame屏幕表面
        """
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.fill(COLOR_BLACK)
        overlay.set_alpha(180)
        screen.blit(overlay, (0, 0))

        if self.use_english:
            pause_text = "Game Paused"
        else:
            pause_text = "游戏暂停"
        text = self.font_large.render(pause_text, True, COLOR_WHITE)
        rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        screen.blit(text, rect)

        if self.use_english:
            hint_text = "Press P to continue"
        else:
            hint_text = "按 P 继续游戏"
        hint = self.font_small.render(hint_text, True, COLOR_GRAY)
        hint_rect = hint.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 60))
        screen.blit(hint, hint_rect)

    def render_victory(self, screen: pygame.Surface):
        """
        渲染胜利界面

        参数:
            screen: Pygame屏幕表面
        """
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.fill(COLOR_BLACK)
        overlay.set_alpha(200)
        screen.blit(overlay, (0, 0))

        if self.use_english:
            victory_text = "Victory!"
        else:
            victory_text = "恭喜胜利！"
        text = self.font_large.render(victory_text, True, COLOR_GOLD)
        rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        screen.blit(text, rect)

        if self.use_english:
            hint_text = "Press R to restart, ESC for menu"
        else:
            hint_text = "按 R 重新开始，按 ESC 返回菜单"
        hint = self.font_small.render(hint_text, True, COLOR_WHITE)
        hint_rect = hint.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 30))
        screen.blit(hint, hint_rect)

    def render_game_over(self, screen: pygame.Surface):
        """
        渲染失败界面

        参数:
            screen: Pygame屏幕表面
        """
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.fill(COLOR_BLACK)
        overlay.set_alpha(200)
        screen.blit(overlay, (0, 0))

        if self.use_english:
            game_over_text = "Game Over"
        else:
            game_over_text = "游戏结束"
        text = self.font_large.render(game_over_text, True, COLOR_RED)
        rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        screen.blit(text, rect)

        if self.use_english:
            hint_text = "Press R to restart, ESC for menu"
        else:
            hint_text = "按 R 重新开始，按 ESC 返回菜单"
        hint = self.font_small.render(hint_text, True, COLOR_WHITE)
        hint_rect = hint.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 30))
        screen.blit(hint, hint_rect)

    def render_help(self, screen: pygame.Surface):
        """
        渲染帮助信息

        参数:
            screen: Pygame屏幕表面
        """
        # 获取当前设置
        difficulty = self.difficulty.capitalize() if self.use_english else \
                    {'easy': '简单', 'normal': '普通', 'hard': '困难'}.get(self.difficulty, '普通')
        ai_mode = "ON" if self.ai_mode else "OFF"
        
        if self.use_english:
            help_lines = [
                f"Controls: | Difficulty: {difficulty} | AI Demo: {ai_mode}",
                "WASD/Arrow - Move | Space - Light | P - Pause | ESC - Menu",
            ]
        else:
            help_lines = [
                f"操作: WASD移动 空格光源 | 难度: {difficulty} | AI演示: {'开' if self.ai_mode else '关'}",
                "P暂停 ESC返回菜单",
            ]

        y_offset = self.screen_height - 100
        for line in help_lines:
            text = self.font_small.render(line, True, COLOR_GRAY)
            screen.blit(text, (self.hud_margin, y_offset))
            y_offset += 25


# =============================================================================
# 第十二部分：游戏管理器
# =============================================================================

class GameManager:
    """
    游戏管理器
    负责游戏状态管理和主循环
    """

    def __init__(self):
        """初始化游戏管理器"""
        # 初始化Pygame
        pygame.init()
        pygame.display.set_caption("开关猎杀 - Switch Hunt")

        # 创建屏幕
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # 游戏状态
        self.state = GameState.MENU

        # 游戏对象
        self.game_map = None
        self.player = None
        self.treasures = []
        self.ghosts = []
        self.visibility_system = None
        self.ui_system = UISystem(SCREEN_WIDTH, SCREEN_HEIGHT)

        # 游戏数据
        self.treasures_collected = 0

        # 相机偏移
        self.camera_offset = (0, 0)

    def init_game(self):
        """初始化游戏对象"""
        # 创建地图
        self.game_map = Map(MAP_WIDTH, MAP_HEIGHT)

        # 创建玩家
        player_pos = self.game_map.get_random_empty_position()
        self.player = Player(
            player_pos[0] * TILE_SIZE + TILE_SIZE // 2,
            player_pos[1] * TILE_SIZE + TILE_SIZE // 2,
            self.game_map
        )

        # 创建宝藏
        self.treasures = []
        for _ in range(TREASURE_COUNT):
            pos = self.game_map.get_random_empty_position()
            while abs(pos[0] - player_pos[0]) < 3 and abs(pos[1] - player_pos[1]) < 3:
                pos = self.game_map.get_random_empty_position()
            self.treasures.append(Treasure(pos[0], pos[1]))

        # 创建鬼（随地图扩大增加数量）
        self.ghosts = []
        for _ in range(4):
            pos = self.game_map.get_random_empty_position()
            while abs(pos[0] - player_pos[0]) < 5 and abs(pos[1] - player_pos[1]) < 5:
                pos = self.game_map.get_random_empty_position()
            self.ghosts.append(Ghost(
                pos[0] * TILE_SIZE + TILE_SIZE // 2,
                pos[1] * TILE_SIZE + TILE_SIZE // 2,
                self.player.speed,
                self.game_map
            ))

        # 创建可见性系统
        self.visibility_system = VisibilitySystem(
            self.game_map, SCREEN_WIDTH, SCREEN_HEIGHT
        )

        # 重置游戏数据
        self.treasures_collected = 0

        # 计算相机偏移（居中地图）
        map_pixel_width = MAP_WIDTH * TILE_SIZE
        map_pixel_height = MAP_HEIGHT * TILE_SIZE
        self.camera_offset = (
            (SCREEN_WIDTH - map_pixel_width) // 2,
            (SCREEN_HEIGHT - map_pixel_height) // 2
        )
        
        # 应用难度设置
        self._apply_difficulty()

    def _apply_difficulty(self):
        """
        根据UI设置的难度应用不同的鬼AI
        """
        difficulty = self.ui_system.difficulty
        
        if difficulty == 'easy':
            # 简单难度：使用基础DQN或A*（较慢的鬼）
            # 这里暂时使用A*，但速度降低
            for ghost in self.ghosts:
                ghost.speed = PLAYER_SPEED * 0.8  # 比玩家慢
                
        elif difficulty == 'hard':
            # 困难难度：使用训练好的高级DQN
            try:
                # 尝试加载DQN模型（如果存在）
                from dqn_model_new import DQNAI
                agent = DQNAI()
                agent.load('models/ghost_hard.pth')
                agent.epsilon = 0  # 纯利用模式
                
                # 替换鬼为DQN控制
                from switch_hunt_dqn_demo import DQNGhost
                new_ghosts = []
                for ghost in self.ghosts:
                    dqn_ghost = DQNGhost(
                        ghost.pos[0], ghost.pos[1],
                        ghost.player_speed, ghost.game_map, agent
                    )
                    new_ghosts.append(dqn_ghost)
                self.ghosts = new_ghosts
                print("[难度] 已加载高级鬼AI (DQN)")
            except:
                # 如果模型不存在，使用更快的A*
                for ghost in self.ghosts:
                    ghost.speed = PLAYER_SPEED * 1.5  # 比玩家快很多
                print("[难度] 未找到DQN模型，使用高速A*")
                
        else:  # normal
            # 普通难度：标准A*
            for ghost in self.ghosts:
                ghost.speed = PLAYER_SPEED * 1.2  # 略快于玩家

    def _get_player_centered_state(self, player, ghost):
        """
        获取以玩家为中心的状态编码（用于AI演示模式）
        
        修改：添加视野限制，玩家只能看到光源范围内的鬼
        普通光源半径3格，强化光源半径4格
        """
        import numpy as np
        
        state = np.zeros((5, 21, 21), dtype=np.float32)
        center_x, center_y = player.get_grid_pos()
        half_size = 10  # 21 // 2
        
        # 计算玩家视野范围（基于光源）
        view_radius = player.light_radius  # 普通3格或强化4格
        
        # 计算鬼是否在视野范围内
        ghost_in_view = False
        dist_to_ghost = np.sqrt((ghost.grid_pos[0] - center_x)**2 + (ghost.grid_pos[1] - center_y)**2)
        if dist_to_ghost <= view_radius:
            ghost_in_view = True
        
        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                grid_x = center_x + dx
                grid_y = center_y + dy
                state_x = dx + half_size
                state_y = dy + half_size
                
                if 0 <= grid_x < MAP_WIDTH and 0 <= grid_y < MAP_HEIGHT:
                    # 通道0: 墙壁（始终可见）
                    if self.game_map.is_wall(grid_x, grid_y):
                        state[0, state_y, state_x] = 1.0
                    
                    # 通道1: 玩家位置（在中心，始终可见）
                    if (grid_x, grid_y) == player.get_grid_pos():
                        state[1, state_y, state_x] = 1.0
                    
                    # 通道2: 鬼位置（视野限制：只在光源范围内可见）
                    if (grid_x, grid_y) == ghost.grid_pos:
                        if ghost_in_view:
                            state[2, state_y, state_x] = 1.0
                        # 否则保持为0（不可见）
                    
                    # 通道3: 光源覆盖区域
                    dist_from_center = np.sqrt(dx**2 + dy**2)
                    if dist_from_center <= view_radius:
                        # 在光源范围内，根据距离衰减
                        state[3, state_y, state_x] = 1.0 - (dist_from_center / view_radius)
                    
                    # 通道4: 光源开启标志（全图统一值）
                    if player.is_enhanced_light():
                        state[4, state_y, state_x] = 1.0
        
        return state

    def run(self):
        """运行游戏主循环"""
        running = True

        while running:
            # 计算时间增量
            dt = self.clock.tick(FPS) / 1000.0

            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    self.handle_event(event)

            # 更新游戏状态
            self.update(dt)

            # 渲染
            self.render()

            # 更新显示
            pygame.display.flip()

        pygame.quit()

    def handle_event(self, event: pygame.event.Event):
        """
        处理事件

        参数:
            event: Pygame事件对象
        """
        if self.state == GameState.MENU:
            action = self.ui_system.handle_menu_input(event)
            if action == "start":
                self.init_game()
                self.state = GameState.PLAYING
            elif action == "quit":
                pygame.quit()
                exit()

        elif self.state == GameState.PLAYING:
            self.player.handle_input(event)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.state = GameState.PAUSED
                elif event.key == pygame.K_ESCAPE:
                    self.state = GameState.MENU
                elif event.key == pygame.K_F1:
                    # F1切换作弊模式
                    self.ui_system.cheat_mode = not self.ui_system.cheat_mode

        elif self.state == GameState.PAUSED:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.state = GameState.PLAYING
                elif event.key == pygame.K_ESCAPE:
                    self.state = GameState.MENU

        elif self.state in (GameState.VICTORY, GameState.GAME_OVER):
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.init_game()
                    self.state = GameState.PLAYING
                elif event.key == pygame.K_ESCAPE:
                    self.state = GameState.MENU

    def update(self, dt: float):
        """
        更新游戏状态

        参数:
            dt: 时间增量（秒）
        """
        if self.state != GameState.PLAYING:
            return

        # AI演示模式：AI自动控制玩家
        if self.ui_system.ai_mode:
            try:
                from dqn_model_new import DQNAI
                # 尝试加载玩家AI模型
                if not hasattr(self, '_player_ai_agent'):
                    self._player_ai_agent = DQNAI()
                    self._player_ai_agent.load('models/player_ai.pth')
                    self._player_ai_agent.epsilon = 0
                
                # 获取AI动作
                ghost = self.ghosts[0]
                state = self._get_player_centered_state(self.player, ghost)
                action = self._player_ai_agent.get_action(state, trainning=False)
                
                # 应用动作到玩家
                self.player.keys_pressed = {
                    'up': False, 'down': False, 'left': False, 'right': False
                }
                actions = ['up', 'down', 'left', 'right']
                if 0 <= action < 4:
                    self.player.keys_pressed[actions[action]] = True
                    
            except Exception as e:
                # AI模型不存在或出错，回退到手动模式
                if hasattr(self, '_ai_demo_error_shown'):
                    pass
                else:
                    print(f"[AI演示] 模型加载失败: {e}")
                    self._ai_demo_error_shown = True

        # 更新玩家
        self.player.update(dt)

        # 更新宝藏
        for treasure in self.treasures:
            treasure.update(dt)
            if treasure.check_pickup(self.player):
                self.treasures_collected += 1
                self.player.add_energy(TREASURE_ENERGY_RESTORE)

        # 更新鬼
        for ghost in self.ghosts:
            ghost.update(dt, self.player)

            # 检测玩家是否使用强化光源定身鬼
            if self.player.is_enhanced_light():
                ghost_pos = (ghost.pos[0], ghost.pos[1])
                player_pos = self.player.get_pixel_pos()
                dist = distance(ghost_pos[0], ghost_pos[1], player_pos[0], player_pos[1])

                light_radius_px = LIGHT_RADIUS_ENHANCED * TILE_SIZE
                if dist <= light_radius_px:
                    ghost.freeze(GHOST_FREEZE_DURATION)

            # 检测鬼与玩家碰撞（游戏失败）
            # 鬼被定身时不会导致游戏失败
            if ghost.state != GhostState.STUNNED and ghost.check_collision(self.player):
                if not self.ui_system.cheat_mode:
                    self.state = GameState.GAME_OVER

        # 更新可见性系统
        self.visibility_system.update(self.player)

        # 检查胜利条件
        if self.treasures_collected >= TREASURE_COUNT:
            self.state = GameState.VICTORY

    def render(self):
        """渲染游戏画面"""
        if self.state == GameState.MENU:
            self.ui_system.render_menu(self.screen)

        elif self.state in (GameState.PLAYING, GameState.PAUSED,
                           GameState.VICTORY, GameState.GAME_OVER):
            # 清空屏幕
            self.screen.fill(COLOR_BLACK)

            # 渲染地图
            self.game_map.render(self.screen, self.camera_offset)

            # 渲染宝藏
            for treasure in self.treasures:
                treasure.render(self.screen, self.camera_offset)

            # 渲染鬼
            for ghost in self.ghosts:
                ghost.render(self.screen, self.camera_offset)

            # 渲染玩家
            self.player.render(self.screen, self.camera_offset)

            # 渲染迷雾（如果不是作弊模式）
            if not self.ui_system.cheat_mode:
                self.visibility_system.render(self.screen, self.player, self.camera_offset)

            # 渲染HUD
            self.ui_system.render_hud(
                self.screen, self.player,
                self.treasures_collected, TREASURE_COUNT
            )

            # 渲染帮助信息
            self.ui_system.render_help(self.screen)

            # 根据状态渲染覆盖层
            if self.state == GameState.PAUSED:
                self.ui_system.render_pause(self.screen)
            elif self.state == GameState.VICTORY:
                self.ui_system.render_victory(self.screen)
            elif self.state == GameState.GAME_OVER:
                self.ui_system.render_game_over(self.screen)


# =============================================================================
# 第十三部分：主程序入口
# =============================================================================

def main():
    """
    主函数
    游戏入口点
    """
    print("=" * 60)
    print("《开关猎杀》- Switch Hunt")
    print("=" * 60)
    print("操作说明:")
    print("  WASD/方向键 - 移动玩家")
    print("  空格键 - 开启/关闭强化光源")
    print("  P键 - 暂停游戏")
    print("  ESC键 - 返回菜单")
    print("  F1键 - 切换作弊模式")
    print("=" * 60)
    print("游戏目标:")
    print("  收集地图上的4个宝藏获胜")
    print("  躲避鬼的追击，被碰到则失败")
    print("  使用强化光源可以定身鬼3秒")
    print("=" * 60)

    # 创建并运行游戏
    game = GameManager()
    game.run()

    print("游戏已退出")


if __name__ == "__main__":
    main()
