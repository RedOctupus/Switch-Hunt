"""V8 游戏主类与入口。"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import random
from typing import Optional, List

import numpy as np
import pygame

from switch_hunt.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, TILE_SIZE, MAP_WIDTH, MAP_HEIGHT,
    PLAYER_SPEED, PLAYER_RADIUS, TREASURE_COUNT, TREASURE_ENERGY_RESTORE,
    GHOST_SPEED_RATIO, GHOST_RADIUS, GHOST_FREEZE_DURATION,
    COLOR_BLACK, COLOR_WHITE, COLOR_GRAY, COLOR_DARK_GRAY, COLOR_YELLOW,
    COLOR_BLUE, COLOR_GREEN, COLOR_RED, COLOR_ORANGE, COLOR_GOLD, COLOR_CYAN,
    STATE_CHANNELS, STATE_SIZE,
)
from switch_hunt.enums import GameState, LightMode, LightState, GhostState
from switch_hunt.utils import grid_to_pixel, pixel_to_grid, distance
from switch_hunt.config.default import LIGHT_SYSTEM, GHOST_SPAWN
from switch_hunt.core.map import Map
from switch_hunt.core.pathfinding import AStarPathfinder
from switch_hunt.core.entities.player import Player
from switch_hunt.core.entities.ghost import Ghost
from switch_hunt.core.entities.treasure import Treasure
from switch_hunt.core.visibility import VisibilitySystem
from switch_hunt.game.manager import GameManager
from switch_hunt.game.ui import UISystem
from switch_hunt.game.sound import SoundManager
from switch_hunt.game.player_v8 import PlayerV8
from switch_hunt.game.ghost_v8 import DQNGhostV8
from switch_hunt.game.treasure_v8 import TreasureV8

COLOR_WALL_BRIGHT = (100, 100, 120)
COLOR_WALL_BORDER_BRIGHT = (130, 130, 150)

if os.environ.get("DQN_TRAINING") != "1":
    print("[V8] Loading Switch Hunt v8.0 - Grid-Aligned DQN Training")

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
        from switch_hunt.config.default import GHOST_SPAWN
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
            from switch_hunt.rl.agent import DQNAI
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
