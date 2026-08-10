"""V8 玩家：光源系统 + 寻宝 AI。"""
from __future__ import annotations

import math
import random
from typing import Optional, List, Tuple

import pygame

from switch_hunt.constants import (
    PLAYER_SPEED, PLAYER_RADIUS, TILE_SIZE, MAP_WIDTH, MAP_HEIGHT,
    TREASURE_ENERGY_RESTORE,
    COLOR_BLUE, COLOR_YELLOW, COLOR_WHITE, COLOR_CYAN, COLOR_GREEN,
)
from switch_hunt.enums import LightMode, LightState, GhostState
from switch_hunt.config.default import LIGHT_SYSTEM
from switch_hunt.utils import grid_to_pixel, pixel_to_grid, distance
from switch_hunt.core.entities.player import Player
from switch_hunt.core.pathfinding import AStarPathfinder

class PlayerV8(Player):
    """V8玩家：完整V7功能 + 光源系统 + A*寻路AI"""
    
    def __init__(self, x, y, game_map):
        super().__init__(x, y, game_map)
        
        # 光源系统（V7）
        self.light_charges = LIGHT_SYSTEM['initial_charges']
        self.light_charges_max = LIGHT_SYSTEM['max_charges']
        self.light_state = LightState.IDLE
        self.light_active_timer = 0.0
        self.light_cooldown_timer = 0.0
        self.light_mode = LightMode.NORMAL
        self.light_radius = LIGHT_SYSTEM['radius']
        
        # V8.11: 反应延迟机制 - 模拟真实玩家反应时间
        # 人类反应时间: 200-400ms (12-24帧 @ 60fps)
        self._reaction_timer = 0.0   # 当前反应倒计时
        self._reaction_delay = 0.0   # 本次随机反应时间
        self._threat_ghost = None    # 当前威胁的鬼
        self._is_reacting = False    # 是否处于反应中状态
        self._reaction_type = None   # 触发类型: 'manual' 或 'auto'
        
        # AI系统（V7）
        self.ai_enabled = False
        self._game_ref = None
        self.pathfinder = AStarPathfinder(game_map)
        self.current_path = []
        self.path_update_timer = 0.0
        
        # 脱困系统（V7）
        self._stuck_timer = 0.0
        self._last_pos = None
        self._unstuck_direction = None
        self._unstuck_timer = 0.0
        
        # V8.24调试计数器
        self._debug_counter = 0
    
    def update(self, dt):
        """V8: 更新光源、AI、然后移动"""
        # 更新光源状态机
        self._update_light_mode(dt)
        
        # AI控制
        if self.ai_enabled and self._game_ref:
            if os.environ.get('DQN_TRAINING') == '1' and hasattr(self, '_debug_counter'):
                self._debug_counter += 1
                if self._debug_counter % 60 == 0:  # 每秒打印一次
                    print(f"[PlayerAI Debug] path长度={len(self.current_path) if self.current_path else 0}, "
                          f"pos={self.pos}, stuck={self._stuck_timer:.1f}")
            self._update_ai(dt)
        elif self.ai_enabled and not self._game_ref:
            if os.environ.get('DQN_TRAINING') == '1':
                print(f"[PlayerAI Warning] ai_enabled=True 但 _game_ref=None!")
        
        # 父类更新（移动）
        super().update(dt)
    
    def _update_light_mode(self, dt):
        """V7: 光源状态机"""
        if self.light_state == LightState.ACTIVE:
            self.light_active_timer -= dt
            self.light_mode = LightMode.ENHANCED
            self.light_radius = LIGHT_SYSTEM.get('enhanced_radius', 3)
            
            if self.light_active_timer <= 0:
                self.light_state = LightState.COOLDOWN
                self.light_cooldown_timer = LIGHT_SYSTEM['cooldown_duration']
                self.light_mode = LightMode.NORMAL
                self.light_radius = LIGHT_SYSTEM['radius']
                
        elif self.light_state == LightState.COOLDOWN:
            self.light_cooldown_timer -= dt
            self.light_mode = LightMode.NORMAL
            self.light_radius = LIGHT_SYSTEM['radius']
            
            if self.light_cooldown_timer <= 0:
                self.light_state = LightState.IDLE
                
        else:  # IDLE
            self.light_mode = LightMode.NORMAL
            self.light_radius = LIGHT_SYSTEM['radius']

            # 仅手动开启（按键）—— 自动触发已移除
            # AI演示模式下由 _update_ai() 设置 keys_pressed['enhance']
            should_activate = self.keys_pressed.get('enhance', False)

            # 执行开灯
            if should_activate and self.light_charges > 0:
                self.light_charges -= 1
                self.light_state = LightState.ACTIVE
                self.light_active_timer = LIGHT_SYSTEM['active_duration']
                self.light_mode = LightMode.ENHANCED
                self.light_radius = LIGHT_SYSTEM.get('enhanced_radius', 3)
                # 重置状态
                self._threat_ghost = None
                self._is_reacting = False
                self._reaction_timer = 0.0
                if os.environ.get('DQN_TRAINING') != '1':
                    print(f"[V8] Light activated! Charges left: {self.light_charges}")
    
    def _update_ai(self, dt):
        """V7: 玩家A*AI - 找宝藏，鬼近时开灯"""
        self.path_update_timer += dt
        
        # 脱困模式计时
        if self._unstuck_direction:
            self._unstuck_timer -= dt
            if self._unstuck_timer <= 0:
                self._unstuck_direction = None
        
        # 检测是否卡住
        if self._last_pos:
            move_dist = math.sqrt((self.pos[0]-self._last_pos[0])**2 + 
                                 (self.pos[1]-self._last_pos[1])**2)
            if move_dist < 1.0:
                self._stuck_timer += dt
            else:
                self._stuck_timer = 0
        self._last_pos = list(self.pos)
        
        # 脱困模式
        if self._stuck_timer > 0.3 and not self._unstuck_direction:
            self._stuck_timer = 0
            self._update_ai_path()
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                         (0.7, 0.7), (0.7, -0.7), (-0.7, 0.7), (-0.7, -0.7)]
            best_dir = None
            best_dist = 0
            for dx, dy in directions:
                for dist in [50, 40, 30, 20, 10]:
                    test_x = self.pos[0] + dx * dist
                    test_y = self.pos[1] + dy * dist
                    if self._can_move_to(test_x, test_y):
                        if dist > best_dist:
                            best_dist = dist
                            best_dir = (dx, dy)
                        break
            if best_dir:
                self._unstuck_direction = best_dir
                self._unstuck_timer = 0.8
        
        # 更新路径
        if self.path_update_timer >= 2.0:
            self.path_update_timer = 0
            if not self.current_path or len(self.current_path) <= 1:
                self._update_ai_path()
        
        # 检查到达路径点
        if self.current_path and len(self.current_path) > 1:
            next_grid = self.current_path[1]
            next_px, next_py = grid_to_pixel(next_grid[0], next_grid[1])
            dist_to_next = math.sqrt((self.pos[0]-next_px)**2 + (self.pos[1]-next_py)**2)
            
            if dist_to_next < 10:
                self.current_path.pop(0)
                if len(self.current_path) == 1:
                    self._update_ai_path()
        
        # 跟随路径或脱困
        if self._unstuck_direction:
            dx, dy = self._unstuck_direction
            threshold = 0.1
            self.keys_pressed = {
                'up': dy < -threshold,
                'down': dy > threshold,
                'left': dx < -threshold,
                'right': dx > threshold,
                'enhance': False
            }
        elif self.current_path and len(self.current_path) > 1:
            target_grid = self.current_path[1]
            target_px, target_py = grid_to_pixel(target_grid[0], target_grid[1])
            
            dx = target_px - self.pos[0]
            dy = target_py - self.pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                dx /= dist
                dy /= dist
            
            abs_dx = abs(dx)
            abs_dy = abs(dy)
            
            # 主轴优先
            if abs_dx > abs_dy:
                move_x = 4 if dx > 0 else -4
                if not self._can_move_to(self.pos[0] + move_x, self.pos[1]):
                    self.keys_pressed = {
                        'up': dy < 0,
                        'down': dy > 0,
                        'left': False,
                        'right': False,
                        'enhance': False
                    }
                else:
                    self.keys_pressed = {
                        'up': False,
                        'down': False,
                        'left': dx < 0,
                        'right': dx > 0,
                        'enhance': False
                    }
            else:
                move_y = 4 if dy > 0 else -4
                if not self._can_move_to(self.pos[0], self.pos[1] + move_y):
                    self.keys_pressed = {
                        'up': False,
                        'down': False,
                        'left': dx < 0,
                        'right': dx > 0,
                        'enhance': False
                    }
                else:
                    self.keys_pressed = {
                        'up': dy < 0,
                        'down': dy > 0,
                        'left': False,
                        'right': False,
                        'enhance': False
                    }
        else:
            self._update_ai_path()
            self.keys_pressed = {'up': False, 'down': False, 'left': False, 'right': False, 'enhance': False}
        
        # V8.25: AI开灯逻辑 — 含300ms反应延迟
        # 只在光源空闲且有次数时检测鬼的距离
        ghost_in_range = False
        if (hasattr(self, '_game_ref') and self._game_ref
                and self.light_state == LightState.IDLE
                and self.light_charges > 0):
            for ghost in self._game_ref.ghosts:
                if ghost.state == GhostState.STUNNED:
                    continue
                dist = math.sqrt((self.pos[0] - ghost.pos[0])**2 +
                                 (self.pos[1] - ghost.pos[1])**2)
                if dist / TILE_SIZE <= LIGHT_SYSTEM.get('auto_light_range', 3):
                    ghost_in_range = True
                    break

        if ghost_in_range:
            # 鬼在范围内：若未开始计时则启动，否则递减计时
            if not self._is_reacting:
                self._is_reacting = True
                self._reaction_timer = LIGHT_SYSTEM.get('light_reaction_delay', 0.3)
            else:
                self._reaction_timer -= dt
                if self._reaction_timer <= 0:
                    # 300ms已过，激活光源
                    self._is_reacting = False
                    self.keys_pressed['enhance'] = True
        else:
            # 鬼离开范围或条件不满足：重置反应状态
            self._is_reacting = False
            self._reaction_timer = 0.0
    
    def _update_ai_path(self):
        """AI更新路径：找最近的未收集宝藏"""
        if not hasattr(self, '_game_ref') or not self._game_ref:
            return
        
        game = self._game_ref
        player_grid = self.get_grid_pos()
        
        nearest_treasure = None
        nearest_dist = float('inf')
        
        for treasure in game.treasures:
            if not treasure.collected:
                dist = math.sqrt((player_grid[0]-treasure.grid_pos[0])**2 + 
                               (player_grid[1]-treasure.grid_pos[1])**2)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_treasure = treasure
        
        if nearest_treasure:
            self.current_path = self.pathfinder.find_path(
                player_grid, nearest_treasure.grid_pos
            )
    
    def is_enhanced_light(self):
        """是否激活强化光源"""
        return self.light_state == LightState.ACTIVE

