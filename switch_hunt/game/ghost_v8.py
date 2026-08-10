"""V8 鬼：网格对齐移动 + 7 通道状态编码。"""
from __future__ import annotations

import math
import os
import random
from typing import Optional, List, Tuple

import numpy as np
import pygame

from switch_hunt.constants import (
    TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, GHOST_SPEED_RATIO, PLAYER_SPEED,
    COLOR_GHOST_NORMAL, COLOR_GHOST_STUNNED, COLOR_WHITE, COLOR_CYAN,
    COLOR_BLUE, COLOR_RED, STATE_CHANNELS, STATE_SIZE,
)
from switch_hunt.enums import GhostState, LightState
from switch_hunt.config.default import LIGHT_SYSTEM, GHOST_MOVE
from switch_hunt.utils import grid_to_pixel, pixel_to_grid, distance
from switch_hunt.core.entities.ghost import Ghost
from switch_hunt.core.pathfinding import AStarPathfinder

class DQNGhostV8(Ghost):
    """V8: 网格对齐DQN控制鬼"""
    
    def __init__(self, grid_x, grid_y, game_map, player):
        x, y = grid_to_pixel(grid_x, grid_y)
        super().__init__(x, y, PLAYER_SPEED, game_map)
        
        self.radius = TILE_SIZE // 2  # 16px
        
        self.player = player
        self.dqn_ai = None
        self.use_dqn = False
        
        self.grid_pos = (grid_x, grid_y)
        
        self.is_moving = False
        self.move_progress = 0.0
        self.target_grid = None
        self.current_action = None

        self.current_path = []
        self.planned_direction = None
        self.path_update_counter = 0
        # 强化光源照射累计时间（秒），用于定身判定与变蓝视觉反馈
        self._stun_exposure = 0.0
        # V8.25: 删除sprint_mode，速度恒为玩家1.2倍，sprint只是奖励结构区分
    
    def update_path(self):
        """更新A*路径"""
        if self.player:
            self.current_path = self.pathfinder.find_path(
                self.grid_pos, self.player.get_grid_pos()
            )
            if len(self.current_path) >= 2:
                next_grid = self.current_path[1]
                dx = next_grid[0] - self.grid_pos[0]
                dy = next_grid[1] - self.grid_pos[1]
                direction_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
                self.planned_direction = direction_map.get((dx, dy), None)
            else:
                self.planned_direction = None
    
    def get_state(self):
        """V8.15: 7通道状态编码 - 包含A*路径引导
        
        通道0: 墙壁地图 (1=墙, 0=空地) - 静态环境
        通道1: 鬼位置 (one-hot)
        通道2: 玩家位置 (one-hot)
        通道3: A*路径 (从鬼到玩家的最优路径，值为1.0)
        通道4: 危险区 (定身范围stun_radius=2格，光源激活时=1.0) - V8.23
        通道5: 光源CD状态 (CD中=1.0)
        通道6: 玩家光源次数归一化比例 (charges/max_charges) - V8.23
        
        注意: A*路径作为引导信息，帮助DQN学习更优策略
        """
        import numpy as np
        state = np.zeros((7, 21, 21), dtype=np.float32)
        
        # 通道0: 墙壁地图
        for y in range(21):
            for x in range(21):
                if self.game_map.is_wall(x, y):
                    state[0, y, x] = 1.0
        
        # 通道1: 鬼位置
        gx, gy = self.grid_pos
        if 0 <= gy < 21 and 0 <= gx < 21:
            state[1, gy, gx] = 1.0
        
        # 通道2: 玩家位置
        if self.player:
            px, py = self.player.get_grid_pos()
            if 0 <= py < 21 and 0 <= px < 21:
                state[2, py, px] = 1.0
        
        # 通道3: A*路径 - 引导DQN学习
        if self.current_path and len(self.current_path) > 0:
            for path_pos in self.current_path:
                px, py = path_pos
                if 0 <= px < 21 and 0 <= py < 21:
                    state[3, py, px] = 1.0
        
        # 通道4: 定身危险区（强化光源激活时，与 enhanced_radius 一致）
        # 进入后累计曝光，满 stun_exposure_time 后定身
        if self.player and hasattr(self.player, 'light_state'):
            if self.player.light_state == LightState.ACTIVE:
                px, py = self.player.get_grid_pos()
                stun_r = LIGHT_SYSTEM.get('enhanced_radius', 5)
                for dy in range(-stun_r, stun_r + 1):
                    for dx in range(-stun_r, stun_r + 1):
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < 21 and 0 <= ny < 21:
                            if dx*dx + dy*dy <= stun_r*stun_r:
                                state[4, ny, nx] = 1.0
        
        # 通道5: 光源CD状态
        if self.player and hasattr(self.player, 'light_state'):
            if self.player.light_state == LightState.COOLDOWN:
                state[5, :, :] = 1.0
        
        # 通道6: 玩家光源次数（归一化比例）
        # V8.23: 二进制→归一化，区分0/1/2/3次剩余（区分sprint与stalk阶段）
        if self.player and hasattr(self.player, 'light_charges'):
            max_charges = max(getattr(self.player, 'light_charges_max', 3), 1)
            state[6, :, :] = self.player.light_charges / max_charges
        
        return state
    
    def get_action(self):
        """DQN动作"""
        if not self.use_dqn or not self.dqn_ai:
            return random.randint(0, 3)
        return self.dqn_ai.get_action(self.get_state(), training=False)
    
    def apply_action(self, action, dt):
        """V8: 网格对齐移动"""
        if self.state == GhostState.STUNNED:
            return False
        
        if self.is_moving:
            return self._continue_move(dt)
        
        return self._start_move(action)
    
    def _start_move(self, action):
        """开始向下一个格子移动"""
        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = actions[action]
        
        target_x = self.grid_pos[0] + dx
        target_y = self.grid_pos[1] + dy
        
        if self.game_map.is_wall(target_x, target_y):
            return False
        
        self.is_moving = True
        self.move_progress = 0.0
        self.target_grid = (target_x, target_y)
        self.current_action = action
        
        return True
    
    def _continue_move(self, dt):
        """继续当前移动 - V8.25: 速度恒为玩家1.2倍（GHOST_MOVE['speed']=5.625）"""
        from switch_hunt.config.default import GHOST_MOVE
        # V8.25: 统一使用单一速度，不再区分sprint/stalk（sprint仅是奖励结构区分）
        move_speed = GHOST_MOVE['speed']
        self.move_progress += move_speed * dt
        
        if self.move_progress >= 1.0:
            self.move_progress = 1.0
            self._complete_move()
            return True
        
        start_pixel = grid_to_pixel(self.grid_pos[0], self.grid_pos[1])
        target_pixel = grid_to_pixel(self.target_grid[0], self.target_grid[1])
        
        self.pos[0] = start_pixel[0] + (target_pixel[0] - start_pixel[0]) * self.move_progress
        self.pos[1] = start_pixel[1] + (target_pixel[1] - start_pixel[1]) * self.move_progress
        
        return True
    
    def _complete_move(self):
        """完成移动"""
        self.grid_pos = self.target_grid
        self.pos = list(grid_to_pixel(self.grid_pos[0], self.grid_pos[1]))
        
        self.is_moving = False
        self.move_progress = 0.0
        self.target_grid = None
    
    def update(self, dt, player):
        """V8更新"""
        self.player = player
        self.step_count += 1
        
        self.path_update_counter += 1
        if self.path_update_counter >= 5:
            self.path_update_counter = 0
            self.update_path()
        
        if self.state == GhostState.STUNNED:
            self.stun_timer -= dt
            if self.stun_timer <= 0:
                self.state = GhostState.NORMAL
        
        if self.state == GhostState.NORMAL and not self.is_moving:
            action = self.get_action()
            self.apply_action(action, dt)
        elif self.is_moving:
            self._continue_move(dt)
    
    def update_for_dqn_training(self, dt, player):
        """DQN训练专用更新 - V8修复: 定身时正确恢复"""
        self.player = player
        self.step_count += 1
        
        # 定身状态：只递减timer，不移动
        if self.state == GhostState.STUNNED:
            self.stun_timer -= dt
            if self.stun_timer <= 0:
                self.state = GhostState.NORMAL
                if os.environ.get('DQN_TRAINING') != '1':
                    print(f"[V8] Ghost recovered from stun!")
            return  # 定身时不执行其他更新
        
        # 正常状态：更新路径和移动
        self.path_update_counter += 1
        if self.path_update_counter >= 5:
            self.path_update_counter = 0
            self.update_path()
        
        if self.is_moving:
            self._continue_move(dt)
    
    def get_exposure_ratio(self) -> float:
        """照射进度 0~1：用于变蓝反馈；定身时视为满进度。"""
        if self.state == GhostState.STUNNED:
            return 1.0
        needed = LIGHT_SYSTEM.get('stun_exposure_time', 1.0)
        if needed <= 0:
            return 0.0
        return max(0.0, min(1.0, getattr(self, '_stun_exposure', 0.0) / needed))

    def render(self, screen, camera_offset=(0, 0)):
        """V8: 渲染网格对齐的鬼；照射下按曝光进度由红渐变蓝。"""
        from switch_hunt.game import theme as T

        screen_x = int(self.pos[0] + camera_offset[0])
        screen_y = int(self.pos[1] + camera_offset[1])
        stunned = self.state == GhostState.STUNNED
        t = self.get_exposure_ratio()

        body = T.lerp_color(T.GHOST_BODY, T.GHOST_STUN_BODY, t)
        core = T.lerp_color(T.GHOST_CORE, T.GHOST_STUN_CORE, t)
        outline = T.lerp_color(T.GHOST_OUTLINE, T.ACCENT_STUN, t)

        # 外晕：随照射加深冰蓝感
        T.draw_soft_glow(
            screen, (screen_x, screen_y),
            self.radius + (14 if stunned or t > 0.6 else 10),
            body,
            strength=0.32 + 0.18 * t,
            rings=8,
        )

        pygame.draw.circle(screen, outline, (screen_x, screen_y), self.radius)
        pygame.draw.circle(screen, body, (screen_x, screen_y), self.radius - 2)
        pygame.draw.circle(
            screen, core,
            (screen_x, screen_y - self.radius // 4),
            max(4, self.radius // 3),
        )

        # 眼睛：随照射由暗红转为冰白
        eye_y = screen_y - 2
        eye_dx = self.radius // 3
        eye_r = 2 if stunned else 3
        eye_color = T.lerp_color((40, 10, 10), (220, 240, 255), t)
        pygame.draw.circle(screen, eye_color, (screen_x - eye_dx, eye_y), eye_r)
        pygame.draw.circle(screen, eye_color, (screen_x + eye_dx, eye_y), eye_r)
        if t < 0.45:
            pygame.draw.circle(screen, (255, 220, 180), (screen_x - eye_dx + 1, eye_y - 1), 1)
            pygame.draw.circle(screen, (255, 220, 180), (screen_x + eye_dx + 1, eye_y - 1), 1)

        # 照射进度环：让玩家直观看到定身读条
        if t > 0.0 and not stunned:
            ring_r = self.radius + 6
            # 底环
            pygame.draw.circle(screen, (40, 50, 70), (screen_x, screen_y), ring_r, 2)
            # 进度弧（从顶部顺时针）
            rect = pygame.Rect(screen_x - ring_r, screen_y - ring_r, ring_r * 2, ring_r * 2)
            start = -math.pi / 2
            end = start + t * 2 * math.pi
            pygame.draw.arc(screen, T.ACCENT_STUN, rect, start, end, 3)

        # 定身：冰裂纹
        if stunned:
            for ang in (0.4, 1.2, 2.1, 3.5, 4.8):
                ex = screen_x + int(math.cos(ang) * (self.radius - 3))
                ey = screen_y + int(math.sin(ang) * (self.radius - 3))
                pygame.draw.line(screen, T.GHOST_STUN_CORE, (screen_x, screen_y), (ex, ey), 1)

