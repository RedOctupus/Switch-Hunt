"""鬼实体（V7 基础移动 / 定身）。"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional

import pygame

from switch_hunt.constants import (
    GHOST_RADIUS, GHOST_SPEED_RATIO, PLAYER_SPEED, TILE_SIZE,
    COLOR_GHOST_NORMAL, COLOR_GHOST_STUNNED, COLOR_WHITE, COLOR_YELLOW,
    GHOST_FREEZE_DURATION,
)
from switch_hunt.enums import GhostState
from switch_hunt.utils import grid_to_pixel, pixel_to_grid, distance
from switch_hunt.core.pathfinding import AStarPathfinder
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from switch_hunt.core.map import Map
    from switch_hunt.core.entities.player import Player

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
