"""
《开关猎杀》V8.0 - 网格对齐DQN训练系统（完整游戏版）
================================================================================
V8.0 核心特性:
1. 网格对齐移动（鬼球直径=32px，圆心始终在格子中心）
2. 方向趋势奖励（根据A*路径方向判断，非位置判断）
3. A*路径每5步更新一次（避免频繁变化）
4. 取消距离奖励，专注方向学习
5. 完整的V7功能继承（UI、渲染、玩家A*AI、光源系统）
================================================================================
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pygame
import numpy as np
import math
import random

# 导入基础游戏
from switch_hunt_v7_base import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, TILE_SIZE, MAP_WIDTH, MAP_HEIGHT,
    PLAYER_SPEED, PLAYER_RADIUS, TREASURE_COUNT, TREASURE_ENERGY_RESTORE,
    GHOST_SPEED_RATIO, GHOST_RADIUS, GHOST_FREEZE_DURATION,
    COLOR_BLACK, COLOR_WHITE, COLOR_GRAY, COLOR_DARK_GRAY, COLOR_YELLOW,
    COLOR_BLUE, COLOR_GREEN, COLOR_RED, COLOR_ORANGE, COLOR_GOLD, COLOR_CYAN,
    pixel_to_grid, grid_to_pixel, distance, LightMode,
    Map, Player, Ghost, LightState, Treasure,
    AStarPathfinder, GameManager, GameState, GhostState,
    VisibilitySystem, STATE_CHANNELS, STATE_SIZE
)

from config_v8 import LIGHT_SYSTEM, GHOST_REWARD

# 训练时减少输出
import os
if os.environ.get('DQN_TRAINING') != '1':
    print("[V8] Loading Switch Hunt v8.0 - Grid-Aligned DQN Training")

# 提亮墙壁颜色
COLOR_WALL_BRIGHT = (100, 100, 120)
COLOR_WALL_BORDER_BRIGHT = (130, 130, 150)


class UISystem:
    """UI系统"""
    def __init__(self):
        self.cheat_mode = False


class SoundManager:
    """程序化音效与背景配乐管理器（无需外部音频文件）"""

    SR = 22050  # 采样率

    def __init__(self):
        self.enabled = True
        self.initialized = False
        self._sounds = {}
        self._bg_sound = None
        self._bg_channel = None
        self._tension_timer = 0.0
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.SR, size=-16, channels=2, buffer=1024)
            self._generate_all()
            self._bg_channel = pygame.mixer.Channel(7)
            self.initialized = True
        except Exception as e:
            print(f"[Sound] 初始化失败: {e}")

    # ── 基础工具 ────────────────────────────────────────────────
    def _make(self, wave):
        """float32 [-1,1] → pygame.Sound (stereo int16)"""
        arr = np.clip(wave, -1.0, 1.0)
        arr16 = (arr * 26000).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack([arr16, arr16]))
        return pygame.sndarray.make_sound(stereo)

    def _t(self, dur):
        return np.linspace(0, dur, int(self.SR * dur), False)

    # ── 音效生成 ─────────────────────────────────────────────────
    def _generate_all(self):
        sr = self.SR

        # 1. 光源激活：高频电流裂变感
        t = self._t(0.35)
        env = np.exp(-t * 12)
        w = 0.55 * np.sin(2 * np.pi * 880 * t) * env
        w += 0.25 * np.sin(2 * np.pi * 1760 * t) * env * np.exp(-t * 18)
        w += np.random.uniform(-0.12, 0.12, len(t)) * env
        self._sounds['light'] = self._make(w)

        # 2. 定身：冰冻扫频下降
        t = self._t(0.5)
        freq_sweep = 600 - 400 * t / 0.5
        phase = np.cumsum(2 * np.pi * freq_sweep / sr)
        env = np.exp(-t * 4)
        w = 0.5 * np.sin(phase) * env
        w += 0.15 * np.sin(2 * np.pi * 3000 * t) * np.exp(-t * 25)
        self._sounds['stun'] = self._make(w)

        # 3. 收集宝藏：上行音阶铃声
        t = self._t(0.65)
        freqs = [523, 659, 784, 1047]   # C5 E5 G5 C6
        w = np.zeros(len(t))
        for i, f in enumerate(freqs):
            s = int(i * 0.09 * sr)
            env_t = np.linspace(0, 0.65 - i * 0.09, len(t) - s, False)
            w[s:] += 0.32 * np.sin(2 * np.pi * f * env_t) * np.exp(-env_t * 5)
        self._sounds['treasure'] = self._make(w)

        # 4. 游戏失败：深沉轰鸣
        t = self._t(1.6)
        env = np.exp(-t * 1.8)
        w = 0.5 * np.sin(2 * np.pi * 55 * t) * env
        w += 0.3 * np.sin(2 * np.pi * 80 * t) * env
        w += 0.08 * np.random.uniform(-1, 1, len(t)) * np.exp(-t * 6)
        self._sounds['game_over'] = self._make(w)

        # 5. 胜利：上行琶音
        t = self._t(1.4)
        vfreqs = [392, 523, 659, 784, 1047]  # G4 C5 E5 G5 C6
        w = np.zeros(len(t))
        for i, f in enumerate(vfreqs):
            s = int(i * 0.17 * sr)
            if s >= len(t):
                break
            e = min(len(t), s + int(0.45 * sr))
            et = np.linspace(0, (e - s) / sr, e - s, False)
            w[s:e] += 0.38 * np.sin(2 * np.pi * f * et) * np.exp(-et * 4)
        self._sounds['victory'] = self._make(w)

        # 6. 紧张心跳：两声低频重击
        t = self._t(0.42)
        e1 = np.exp(-t * 28)
        e2 = np.zeros(len(t))
        mid = int(0.19 * sr)
        if mid < len(t):
            e2[mid:] = np.exp(-np.linspace(0, 0.23, len(t) - mid) * 28)
        w = 0.65 * np.sin(2 * np.pi * 58 * t) * e1
        w += 0.55 * np.sin(2 * np.pi * 58 * t) * e2
        self._sounds['heartbeat'] = self._make(w)

        # 7. 背景配乐：暗黑无缝循环（~6秒）
        t = self._t(6.0)
        w = np.zeros(len(t))
        for freq, vol in [(55, 0.28), (82.5, 0.18), (110, 0.11), (73.4, 0.08)]:
            wobble = 0.6 * np.sin(2 * np.pi * 0.22 * t)
            w += vol * np.sin(2 * np.pi * freq * t + wobble)
        # 缓慢震颤 LFO
        w *= 0.58 + 0.42 * np.sin(2 * np.pi * 0.14 * t)
        # 低频脉冲纹理
        pulse = np.maximum(0, np.sin(2 * np.pi * 0.75 * t)) ** 4
        w += 0.10 * np.sin(2 * np.pi * 55 * t) * pulse
        w /= (np.max(np.abs(w)) + 1e-8)
        w *= 0.38
        self._bg_sound = self._make(w)

    # ── 公共接口 ──────────────────────────────────────────────────
    def play(self, name, volume=1.0):
        if not self.enabled or not self.initialized:
            return
        s = self._sounds.get(name)
        if s:
            s.set_volume(volume)
            s.play()

    def start_music(self):
        if self.initialized and self._bg_sound and self._bg_channel and self.enabled:
            self._bg_channel.set_volume(0.45)
            self._bg_channel.play(self._bg_sound, loops=-1)

    def stop_music(self):
        if self.initialized and self._bg_channel:
            self._bg_channel.stop()

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.start_music()
        else:
            self.stop_music()

    def update(self, dt, game):
        """根据鬼的距离动态触发心跳紧张音效"""
        if not self.enabled or not self.initialized:
            return
        self._tension_timer -= dt
        if self._tension_timer > 0 or not game.ghosts:
            return
        player = game.player
        closest = float('inf')
        for ghost in game.ghosts:
            if ghost.state != GhostState.STUNNED:
                d = math.sqrt((player.pos[0] - ghost.pos[0])**2 +
                              (player.pos[1] - ghost.pos[1])**2)
                closest = min(closest, d)
        danger_range = 5 * TILE_SIZE
        if closest <= danger_range:
            self.play('heartbeat', volume=0.6 + 0.4 * (1 - closest / danger_range))
            # 越近心跳越快
            self._tension_timer = 0.35 + (closest / danger_range) * 0.75
        else:
            self._tension_timer = 0.5


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
        
        # 通道4: 定身危险区（光源激活时的完整定身检测范围）
        # V8.25: 改用 enhanced_radius（3格）= 与光源范围一致，进入后1秒定身
        if self.player and hasattr(self.player, 'light_state'):
            if self.player.light_state == LightState.ACTIVE:
                px, py = self.player.get_grid_pos()
                stun_r = LIGHT_SYSTEM.get('enhanced_radius', 3)
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
        from config_v8 import GHOST_MOVE
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
    
    def render(self, screen, camera_offset=(0, 0)):
        """V8: 渲染网格对齐的鬼"""
        screen_x = int(self.pos[0] + camera_offset[0])
        screen_y = int(self.pos[1] + camera_offset[1])
        
        if self.state == GhostState.STUNNED:
            color = COLOR_BLUE
        else:
            color = COLOR_RED
        
        pygame.draw.circle(screen, color, (screen_x, screen_y), self.radius)
        pygame.draw.circle(screen, COLOR_WHITE, (screen_x, screen_y), self.radius, 2)
        
        if self.use_dqn:
            font = pygame.font.Font(None, 16)
            text = font.render("DQN", True, COLOR_WHITE)
            screen.blit(text, (screen_x - 10, screen_y - 25))
        
        if self.planned_direction is not None:
            arrow_length = 20
            if self.planned_direction == 0:
                end_y = screen_y - arrow_length
                pygame.draw.line(screen, (0, 255, 0), (screen_x, screen_y), (screen_x, end_y), 3)
            elif self.planned_direction == 1:
                end_y = screen_y + arrow_length
                pygame.draw.line(screen, (0, 255, 0), (screen_x, screen_y), (screen_x, end_y), 3)
            elif self.planned_direction == 2:
                end_x = screen_x - arrow_length
                pygame.draw.line(screen, (0, 255, 0), (screen_x, screen_y), (end_x, screen_y), 3)
            elif self.planned_direction == 3:
                end_x = screen_x + arrow_length
                pygame.draw.line(screen, (0, 255, 0), (screen_x, screen_y), (end_x, screen_y), 3)


class TreasureV8(Treasure):
    """V8宝藏"""
    def __init__(self, grid_x, grid_y):
        super().__init__(grid_x, grid_y)


class GameV8(GameManager):
    """V8游戏管理器 - 完整游戏功能"""
    
    def __init__(self):
        super().__init__()
        pygame.display.set_caption("开关猎杀 v8.0 - 网格对齐DQN训练")
        
        self.ui_system = UISystem()
        self.player_ai_enabled = False
        self.menu_selected = 0
        self.menu_options = ["开始游戏", "作弊模式", "AI演示模式", "音效/配乐", "退出游戏"]
        self.show_ghost_path = False

        self.treasures_collected = 0
        self.camera_offset = (0, 0)

        self.sound = SoundManager()
        # 用于检测状态跳变以触发一次性音效
        self._prev_game_state = self.state
        self._prev_light_state = None
        self._prev_stunned = set()
    
    def _get_valid_ghost_spawn(self, player_grid):
        """V8.25: 获取距离玩家适当距离（A*步数在[min,max]范围）的鬼出生位置。
        避免鬼出生太近（被轻易引诱）或太远（训练效率低）。
        """
        from config_v8 import GHOST_SPAWN
        min_steps = GHOST_SPAWN.get('min_steps', 7)
        max_steps = GHOST_SPAWN.get('max_steps', 12)
        max_attempts = GHOST_SPAWN.get('max_attempts', 50)

        pathfinder = AStarPathfinder(self.game_map)

        for _ in range(max_attempts):
            gx, gy = self.game_map.get_random_empty_position()
            if (gx, gy) == player_grid:
                continue
            path = pathfinder.find_path((gx, gy), player_grid)
            if path and min_steps <= len(path) - 1 <= max_steps:
                return gx, gy

        # 回退：尝试找任意不与玩家重叠的位置
        for _ in range(20):
            gx, gy = self.game_map.get_random_empty_position()
            if (gx, gy) != player_grid:
                return gx, gy
        return self.game_map.get_random_empty_position()

    def init_game(self):
        """V8初始化"""
        self.game_map = Map(MAP_WIDTH, MAP_HEIGHT)

        start_x, start_y = self.game_map.get_random_empty_position()
        px, py = grid_to_pixel(start_x, start_y)
        self.player = PlayerV8(px, py, self.game_map)
        self.player._game_ref = self
        self.player.ai_enabled = self.player_ai_enabled

        self.treasures = []
        for _ in range(TREASURE_COUNT):
            tx, ty = self.game_map.get_random_empty_position()
            self.treasures.append(TreasureV8(tx, ty))

        # V8.25: 鬼出生位置距玩家适当距离（A*步数 min_steps~max_steps）
        player_grid = (start_x, start_y)
        gx, gy = self._get_valid_ghost_spawn(player_grid)
        ghost = DQNGhostV8(gx, gy, self.game_map, self.player)
        
        # V8.20修复: 改进模型加载逻辑，处理通道不匹配问题
        ghost.use_dqn = False  # 默认不使用DQN，除非成功加载
        try:
            from dqn_model_v8 import DQNAI
            model_paths = ['models/ghost_v8.pth', '../models/ghost_v8.pth', '../../models/ghost_v8.pth']
            for model_path in model_paths:
                if os.path.exists(model_path):
                    # 先尝试加载为7通道模型
                    ghost.dqn_ai = DQNAI(state_channels=7, epsilon=0.0)
                    try:
                        ghost.dqn_ai.load(model_path)
                        # 检查是否成功加载权重（通过检查Qnet的权重是否仍然是初始值）
                        # 如果load方法因通道不匹配而返回，权重不会被加载
                        # 我们假设如果到这里没有异常，说明加载成功
                        ghost.use_dqn = True
                        if os.environ.get('DQN_TRAINING') != '1':
                            print(f"[V8] Ghost model loaded from {model_path}")
                        break
                    except Exception as load_err:
                        # 加载失败，尝试6通道
                        if os.environ.get('DQN_TRAINING') != '1':
                            print(f"[V8] 7通道模型加载失败，尝试6通道: {load_err}")
                        ghost.dqn_ai = DQNAI(state_channels=6, epsilon=0.0)
                        try:
                            ghost.dqn_ai.load(model_path)
                            ghost.use_dqn = True
                            if os.environ.get('DQN_TRAINING') != '1':
                                print(f"[V8] Ghost model (6通道) loaded from {model_path}")
                            break
                        except Exception as load_err2:
                            if os.environ.get('DQN_TRAINING') != '1':
                                print(f"[V8] 6通道模型也加载失败: {load_err2}")
                            ghost.dqn_ai = None
        except Exception as e:
            if os.environ.get('DQN_TRAINING') != '1':
                print(f"[V8] No model found: {e}")
        
        # V8.20: 如果模型未加载成功，确保使用随机动作
        if not ghost.use_dqn:
            if os.environ.get('DQN_TRAINING') != '1':
                print("[V8] DQN模型未加载，鬼将使用随机动作")
        
        self.ghosts = [ghost]
        
        self.visibility_system = VisibilitySystem(self.game_map, SCREEN_WIDTH, SCREEN_HEIGHT)
        
        map_pixel_width = MAP_WIDTH * TILE_SIZE
        map_pixel_height = MAP_HEIGHT * TILE_SIZE
        self.camera_offset = (
            (SCREEN_WIDTH - map_pixel_width) // 2,
            (SCREEN_HEIGHT - map_pixel_height) // 2
        )
        
        self.treasures_collected = 0
        self.state = GameState.PLAYING
        self._prev_game_state = GameState.PLAYING
        self._prev_light_state = None
        self._prev_stunned = set()
        self.sound.stop_music()
        self.sound.start_music()
    
    def _check_treasures(self):
        """V8: 检查宝藏收集 - 增大检测半径确保经过能吃到"""
        for treasure in self.treasures[:]:
            if not treasure.collected:
                # 使用玩家中心位置计算距离
                player_center = self.player.get_pixel_pos()
                dist = math.sqrt((player_center[0]-treasure.pixel_pos[0])**2 + 
                               (player_center[1]-treasure.pixel_pos[1])**2)
                # 增大检测半径（增加10像素容差）
                pickup_radius = PLAYER_RADIUS + treasure.radius + 10
                if dist < pickup_radius:
                    treasure.collected = True
                    self.treasures.remove(treasure)
                    self.treasures_collected += 1
                    self.sound.play('treasure')
                    if isinstance(self.player, PlayerV8):
                        self.player.light_charges = self.player.light_charges_max
                        if os.environ.get('DQN_TRAINING') != '1':
                            print(f"[V8] 宝藏! 光源次数重置为 {self.player.light_charges_max}")
    
    def _update_light_stun(self, dt=0.0):
        """V8.25: 光源定身鬼 — 训练和游戏使用统一规则。
        鬼进入 enhanced_radius（3格）后持续曝光 stun_exposure_time（1秒）才定身。
        玩家进入光源范围即被发现，不存在比光源更小的"定身半径"。
        """
        stun_r_px = LIGHT_SYSTEM.get('enhanced_radius', 3) * TILE_SIZE
        exposure_needed = LIGHT_SYSTEM.get('stun_exposure_time', 1.0)

        if isinstance(self.player, PlayerV8) and self.player.light_state == LightState.ACTIVE:
            for ghost in self.ghosts:
                if ghost.state == GhostState.STUNNED:
                    ghost._stun_exposure = 0.0
                    continue
                dist = distance(self.player.pos[0], self.player.pos[1],
                                ghost.pos[0], ghost.pos[1])
                if dist <= stun_r_px:
                    ghost._stun_exposure = getattr(ghost, '_stun_exposure', 0.0) + dt
                    if ghost._stun_exposure >= exposure_needed:
                        ghost._stun_exposure = 0.0
                        ghost.freeze(LIGHT_SYSTEM['stun_duration'] / 1000.0)
                else:
                    ghost._stun_exposure = 0.0  # 离开范围则重置曝光计时
        else:
            # 光源未激活：清空所有曝光计时
            for ghost in self.ghosts:
                ghost._stun_exposure = 0.0
    
    def _check_game_over(self):
        """V8: 检查游戏结束"""
        if len(self.treasures) == 0:
            self.state = GameState.VICTORY
            return
        
        for ghost in self.ghosts:
            if ghost.state == GhostState.STUNNED:
                continue
            dist = distance(self.player.pos[0], self.player.pos[1],
                          ghost.pos[0], ghost.pos[1])
            if dist < (PLAYER_RADIUS + ghost.radius):
                self.state = GameState.GAME_OVER
                return
    
    def run(self):
        """V8: 主循环"""
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if not self._handle_keydown(event.key):
                        running = False
            
            if self.state == GameState.PLAYING:
                self._update_playing(dt)
            
            self._render()
            pygame.display.flip()

        self.sound.stop_music()
        pygame.quit()
    
    def _handle_keydown(self, key):
        """处理按键"""
        if self.state == GameState.MENU:
            if key == pygame.K_UP:
                self.menu_selected = (self.menu_selected - 1) % len(self.menu_options)
            elif key == pygame.K_DOWN:
                self.menu_selected = (self.menu_selected + 1) % len(self.menu_options)
            elif key == pygame.K_RETURN:
                if self.menu_selected == 0:
                    self.init_game()
                elif self.menu_selected == 1:
                    self.ui_system.cheat_mode = not self.ui_system.cheat_mode
                    print(f"[V8] 作弊模式: {'开启' if self.ui_system.cheat_mode else '关闭'}")
                elif self.menu_selected == 2:
                    self.player_ai_enabled = not self.player_ai_enabled
                    if self.player is not None:
                        self.player.ai_enabled = self.player_ai_enabled
                    if os.environ.get('DQN_TRAINING') != '1':
                        print(f"[V8] AI演示模式: {'开启' if self.player_ai_enabled else '关闭'}")
                elif self.menu_selected == 3:
                    self.sound.toggle()
                    print(f"[V8] 音效/配乐: {'开启' if self.sound.enabled else '关闭'}")
                elif self.menu_selected == 4:
                    return False
            elif key == pygame.K_ESCAPE:
                self.sound.stop_music()
                return False
            elif key == pygame.K_F1:
                self.ui_system.cheat_mode = not self.ui_system.cheat_mode
                if os.environ.get('DQN_TRAINING') != '1':
                    print(f"[V8] 作弊模式: {'开启' if self.ui_system.cheat_mode else '关闭'}")
            elif key == pygame.K_F2:
                self.player_ai_enabled = not self.player_ai_enabled
                if self.player is not None:
                    self.player.ai_enabled = self.player_ai_enabled
                if os.environ.get('DQN_TRAINING') != '1':
                    print(f"[V8] AI演示模式: {'开启' if self.player_ai_enabled else '关闭'}")
            elif key == pygame.K_F4:
                self.sound.toggle()
                print(f"[V8] 音效/配乐: {'开启' if self.sound.enabled else '关闭'}")
        
        elif self.state == GameState.PLAYING:
            if key == pygame.K_ESCAPE:
                self.state = GameState.PAUSED
            elif key == pygame.K_SPACE:
                if isinstance(self.player, PlayerV8):
                    p = self.player
                    if p.light_state == LightState.IDLE and p.light_charges > 0:
                        p.keys_pressed['enhance'] = True
                    elif p.light_state == LightState.COOLDOWN:
                        print("[V8] 光源冷却中!")
                    elif p.light_charges == 0:
                        print("[V8] 光源次数不足!")
            elif key == pygame.K_p:
                self.state = GameState.PAUSED
            elif key == pygame.K_F2:
                self.player.ai_enabled = not self.player.ai_enabled
                print(f"[V8] 玩家AI: {'开启' if self.player.ai_enabled else '关闭'}")
            elif key == pygame.K_F3:
                self.show_ghost_path = not self.show_ghost_path
                print(f"[V8] 鬼A*路径显示: {'开启' if self.show_ghost_path else '关闭'}")
            elif key == pygame.K_F4:
                self.sound.toggle()
                print(f"[V8] 音效/配乐: {'开启' if self.sound.enabled else '关闭'}")

        elif self.state == GameState.PAUSED:
            if key in (pygame.K_p, pygame.K_ESCAPE):
                self.state = GameState.PLAYING
            elif key == pygame.K_RETURN:
                self.state = GameState.MENU
        
        elif self.state in (GameState.GAME_OVER, GameState.VICTORY):
            if key == pygame.K_RETURN:
                self.init_game()
            elif key == pygame.K_ESCAPE:
                self.sound.stop_music()
                self.state = GameState.MENU
        
        return True
    
    def _update_playing(self, dt):
        """更新游戏状态"""
        if not getattr(self.player, 'ai_enabled', False):
            keys = pygame.key.get_pressed()
            self.player.keys_pressed = {
                'up': keys[pygame.K_w] or keys[pygame.K_UP],
                'down': keys[pygame.K_s] or keys[pygame.K_DOWN],
                'left': keys[pygame.K_a] or keys[pygame.K_LEFT],
                'right': keys[pygame.K_d] or keys[pygame.K_RIGHT],
                'enhance': keys[pygame.K_SPACE]
            }

        # 记录更新前的光源状态，用于检测跳变
        prev_ls = self._prev_light_state
        prev_stunned = self._prev_stunned.copy()

        self.player.update(dt)

        # 光源激活音效（IDLE→ACTIVE）
        cur_ls = self.player.light_state if isinstance(self.player, PlayerV8) else None
        if cur_ls == LightState.ACTIVE and prev_ls != LightState.ACTIVE:
            self.sound.play('light')
        self._prev_light_state = cur_ls

        for ghost in self.ghosts:
            ghost.update(dt, self.player)

        self.visibility_system.update(self.player)

        # 定身音效：记录定身前后各鬼的状态
        self._update_light_stun(dt)
        cur_stunned = {id(g) for g in self.ghosts if g.state == GhostState.STUNNED}
        for g in self.ghosts:
            if g.state == GhostState.STUNNED and id(g) not in prev_stunned:
                self.sound.play('stun')
                break
        self._prev_stunned = cur_stunned

        self._check_treasures()

        # 游戏结束/胜利音效（状态跳变一次性触发）
        prev_gs = self._prev_game_state
        self._check_game_over()
        if self.state != prev_gs:
            if self.state == GameState.GAME_OVER:
                self.sound.stop_music()
                self.sound.play('game_over')
            elif self.state == GameState.VICTORY:
                self.sound.stop_music()
                self.sound.play('victory')
        self._prev_game_state = self.state

        # 心跳紧张音效（距离感知）
        self.sound.update(dt, self)
    
    def _render(self):
        """渲染"""
        if self.state == GameState.MENU:
            self._render_menu_v8()
        elif self.state in (GameState.PLAYING, GameState.PAUSED):
            self._render_v8()
            self._render_hud_v8()
            if self.state == GameState.PAUSED:
                self._render_pause_v8()
        elif self.state == GameState.GAME_OVER:
            self._render_v8()
            self._render_game_over_v8()
        elif self.state == GameState.VICTORY:
            self._render_v8()
            self._render_victory_v8()
    
    def _render_v8(self):
        """V8: 游戏画面渲染"""
        self.screen.fill(COLOR_BLACK)
        
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                rect = pygame.Rect(
                    x * TILE_SIZE + self.camera_offset[0],
                    y * TILE_SIZE + self.camera_offset[1],
                    TILE_SIZE, TILE_SIZE
                )
                if self.game_map.is_wall(x, y):
                    pygame.draw.rect(self.screen, COLOR_WALL_BRIGHT, rect)
                    pygame.draw.rect(self.screen, COLOR_WALL_BORDER_BRIGHT, rect, 2)
                else:
                    pygame.draw.rect(self.screen, (40, 40, 50), rect)
        
        for treasure in self.treasures:
            treasure.render(self.screen, self.camera_offset)
        
        for ghost in self.ghosts:
            # [V8.26] 鬼只要进入光源范围（视野半径）就显形，不限于强化状态
            if getattr(self.ui_system, 'cheat_mode', False):
                ghost_visible = True
            elif isinstance(self.player, PlayerV8):
                dist = math.sqrt((ghost.pos[0] - self.player.pos[0])**2 +
                                 (ghost.pos[1] - self.player.pos[1])**2)
                ghost_visible = dist <= self.player.light_radius * TILE_SIZE
            else:
                ghost_visible = False
            if ghost_visible:
                ghost.render(self.screen, self.camera_offset)

        if self.show_ghost_path:
            self._render_ghost_paths()
        
        self.player.render(self.screen, self.camera_offset)
        
        if not getattr(self.ui_system, 'cheat_mode', False):
            self._render_fog_v8()
    
    def _render_ghost_paths(self):
        """渲染鬼的A*路径"""
        for ghost in self.ghosts:
            if not ghost.current_path:
                continue
            
            path_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            
            for grid_x, grid_y in ghost.current_path:
                rect = pygame.Rect(
                    grid_x * TILE_SIZE + self.camera_offset[0] + 8,
                    grid_y * TILE_SIZE + self.camera_offset[1] + 8,
                    TILE_SIZE - 16, TILE_SIZE - 16
                )
                pygame.draw.rect(path_surface, (0, 255, 255, 128), rect, border_radius=4)
            
            if len(ghost.current_path) > 1:
                points = []
                for grid_x, grid_y in ghost.current_path:
                    px = grid_x * TILE_SIZE + TILE_SIZE // 2 + self.camera_offset[0]
                    py = grid_y * TILE_SIZE + TILE_SIZE // 2 + self.camera_offset[1]
                    points.append((px, py))
                
                if len(points) > 1:
                    pygame.draw.lines(path_surface, (255, 255, 255, 100), False, points, 2)
            
            if ghost.current_path:
                target = ghost.current_path[-1]
                tx = target[0] * TILE_SIZE + TILE_SIZE // 2 + self.camera_offset[0]
                ty = target[1] * TILE_SIZE + TILE_SIZE // 2 + self.camera_offset[1]
                pygame.draw.circle(path_surface, (255, 0, 0, 180), (tx, ty), 6)
                pygame.draw.circle(path_surface, (255, 255, 255, 200), (tx, ty), 6, 2)
            
            self.screen.blit(path_surface, (0, 0))
    
    def _render_fog_v8(self):
        """V8: 渲染迷雾"""
        fog_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        fog_surface.fill((0, 0, 0, 0))
        
        player_pos = self.player.get_pixel_pos()
        light_radius_px = self.player.light_radius * TILE_SIZE
        center_x = int(player_pos[0] + self.camera_offset[0])
        center_y = int(player_pos[1] + self.camera_offset[1])
        
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                rect = pygame.Rect(
                    x * TILE_SIZE + self.camera_offset[0],
                    y * TILE_SIZE + self.camera_offset[1],
                    TILE_SIZE, TILE_SIZE
                )
                
                grid_pixel_x = x * TILE_SIZE + TILE_SIZE // 2 + self.camera_offset[0]
                grid_pixel_y = y * TILE_SIZE + TILE_SIZE // 2 + self.camera_offset[1]
                dist = math.sqrt((grid_pixel_x - center_x)**2 + (grid_pixel_y - center_y)**2)
                
                in_light = dist <= light_radius_px
                is_explored = self.visibility_system.explored[y][x]
                
                if in_light:
                    darkness = int(100 * (dist / light_radius_px))
                    darkness = max(0, min(40, darkness))
                    pygame.draw.rect(fog_surface, (0, 0, 0, darkness), rect)
                elif is_explored:
                    pygame.draw.rect(fog_surface, (0, 0, 0, 120), rect)
                else:
                    pygame.draw.rect(fog_surface, (0, 0, 0, 250), rect)
        
        self.screen.blit(fog_surface, (0, 0))
    
    def _render_menu_v8(self):
        """V8: 中文菜单"""
        self.screen.fill(COLOR_BLACK)
        
        try:
            font_large = pygame.font.SysFont("simhei", 64)
            font_medium = pygame.font.SysFont("simhei", 36)
            font_small = pygame.font.SysFont("simhei", 28)
            font_tiny = pygame.font.SysFont("simhei", 22)
        except:
            font_large = pygame.font.Font(None, 64)
            font_medium = pygame.font.Font(None, 36)
            font_small = pygame.font.Font(None, 28)
            font_tiny = pygame.font.Font(None, 22)
        
        title = font_large.render("开关猎杀 v8.0", True, COLOR_GOLD)
        subtitle = font_medium.render("网格对齐DQN训练", True, COLOR_ORANGE)
        
        self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 150))
        self.screen.blit(subtitle, (SCREEN_WIDTH//2 - subtitle.get_width()//2, 220))
        
        option_y_start = 320
        option_spacing = 50
        
        for i, option in enumerate(self.menu_options):
            y = option_y_start + i * option_spacing
            
            if i == self.menu_selected:
                color = COLOR_YELLOW
                prefix = "> "
                pygame.draw.rect(self.screen, (50, 50, 70), 
                               (SCREEN_WIDTH//2 - 200, y - 10, 400, 40), border_radius=5)
            else:
                color = COLOR_WHITE
                prefix = "  "
            
            if i == 1:
                status = "[开启]" if self.ui_system.cheat_mode else "[关闭]"
                text = font_small.render(f"{prefix}{option} {status}", True,
                                        COLOR_GREEN if self.ui_system.cheat_mode else color)
            elif i == 2:
                status = "[开启]" if self.player_ai_enabled else "[关闭]"
                text = font_small.render(f"{prefix}{option} {status}", True,
                                        COLOR_CYAN if self.player_ai_enabled else color)
            elif i == 3:
                snd_on = self.sound.enabled
                status = "[开启]" if snd_on else "[关闭]"
                text = font_small.render(f"{prefix}{option} {status}", True,
                                        COLOR_ORANGE if snd_on else color)
            else:
                text = font_small.render(f"{prefix}{option}", True, color)

            self.screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, y))

        hint_text = font_tiny.render("上下选择 | Enter确认 | F1作弊 | F2AI | F4音效 | ESC退出", True, COLOR_GRAY)
        self.screen.blit(hint_text, (SCREEN_WIDTH//2 - hint_text.get_width()//2, 550))
    
    def _render_hud_v8(self):
        """V8: 中文HUD"""
        try:
            font = pygame.font.SysFont("simhei", 24)
            font_small = pygame.font.SysFont("simhei", 20)
        except:
            font = pygame.font.Font(None, 24)
            font_small = pygame.font.Font(None, 20)
        
        treasure_text = f"宝藏: {self.treasures_collected}/{TREASURE_COUNT}"
        self.screen.blit(font.render(treasure_text, True, COLOR_GOLD), (20, 20))
        
        if isinstance(self.player, PlayerV8):
            p = self.player
            if p.light_state == LightState.ACTIVE:
                text = f"光源: 激活中 {p.light_active_timer:.1f}秒"
                color = COLOR_ORANGE
            elif p.light_state == LightState.COOLDOWN:
                text = f"光源: 冷却中 {p.light_cooldown_timer:.1f}秒"
                color = COLOR_YELLOW
            else:
                text = f"光源次数: {p.light_charges}/{p.light_charges_max}"
                color = COLOR_GREEN if p.light_charges > 0 else COLOR_RED
            
            self.screen.blit(font.render(text, True, color), (20, 50))
            
            if p.ai_enabled:
                ai_text = font.render("[AI自动]", True, COLOR_CYAN)
                self.screen.blit(ai_text, (20, 80))
        
        if self.ui_system.cheat_mode:
            cheat_text = font_small.render("[作弊模式]", True, COLOR_RED)
            self.screen.blit(cheat_text, (SCREEN_WIDTH - 120, 20))
    
    def _render_pause_v8(self):
        """V8: 中文暂停界面"""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        try:
            font = pygame.font.SysFont("simhei", 48)
            font_small = pygame.font.SysFont("simhei", 28)
        except:
            font = pygame.font.Font(None, 48)
            font_small = pygame.font.Font(None, 28)
        
        text = font.render("游戏暂停", True, COLOR_WHITE)
        self.screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 - 30))
        
        hint = font_small.render("P/ESC继续 | Enter返回菜单", True, COLOR_GRAY)
        self.screen.blit(hint, (SCREEN_WIDTH//2 - hint.get_width()//2, SCREEN_HEIGHT//2 + 30))
    
    def _render_game_over_v8(self):
        """V8: 中文游戏结束"""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        try:
            font = pygame.font.SysFont("simhei", 56)
            font_small = pygame.font.SysFont("simhei", 28)
        except:
            font = pygame.font.Font(None, 56)
            font_small = pygame.font.Font(None, 28)
        
        text = font.render("游戏失败", True, COLOR_RED)
        self.screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 - 40))
        
        hint = font_small.render("Enter重新开始 | ESC返回菜单", True, COLOR_WHITE)
        self.screen.blit(hint, (SCREEN_WIDTH//2 - hint.get_width()//2, SCREEN_HEIGHT//2 + 20))
    
    def _render_victory_v8(self):
        """V8: 中文胜利界面"""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        try:
            font = pygame.font.SysFont("simhei", 56)
            font_small = pygame.font.SysFont("simhei", 28)
        except:
            font = pygame.font.Font(None, 56)
            font_small = pygame.font.Font(None, 28)
        
        text = font.render("恭喜通关!", True, COLOR_GOLD)
        self.screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 - 40))
        
        hint = font_small.render("Enter重新开始 | ESC返回菜单", True, COLOR_WHITE)
        self.screen.blit(hint, (SCREEN_WIDTH//2 - hint.get_width()//2, SCREEN_HEIGHT//2 + 20))


def main():
    """V8主函数"""
    print("=" * 60)
    print("开关猎杀 v8.0 - 网格对齐DQN训练")
    print("=" * 60)
    print("操作说明:")
    print("  WASD/方向键: 移动")
    print("  空格: 手动激活强化光源（需自行判断时机）")
    print("  P: 暂停 | ESC: 菜单")
    print("  F1: 作弊模式 | F2: AI演示 | F3: 显示A*路径 | F4: 音效/配乐")
    print("提示: 鬼只在光源激活时可见，保存光源次数很重要！")
    print("=" * 60)
    
    game = GameV8()
    game.run()


if __name__ == "__main__":
    main()
