"""玩家实体与光源状态机。"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional

import pygame

from switch_hunt.constants import (
    PLAYER_RADIUS, PLAYER_SPEED, PLAYER_MAX_ENERGY,
    PLAYER_ENERGY_DECAY, PLAYER_ENERGY_REGEN,
    TILE_SIZE, MAP_WIDTH, MAP_HEIGHT,
    LIGHT_RADIUS_NORMAL, LIGHT_RADIUS_ENHANCED,
    COLOR_BLUE, COLOR_YELLOW, COLOR_WHITE, COLOR_CYAN, COLOR_ORANGE,
)
from switch_hunt.enums import LightMode, LightState
from switch_hunt.config.default import LIGHT_SYSTEM
from switch_hunt.utils import grid_to_pixel, pixel_to_grid, distance, clamp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from switch_hunt.core.map import Map

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
