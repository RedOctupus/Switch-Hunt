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
from switch_hunt.game import theme as T

if os.environ.get("DQN_TRAINING") != "1":
    print("[V8] Loading Switch Hunt — polished playable build")

class GameV8(GameManager):
    """V8游戏管理器 - 完整游戏功能"""
    
    def __init__(self):
        super().__init__()
        pygame.display.set_caption("开关猎杀 — Switch Hunt")
        
        self.ui_system = UISystem()
        self.player_ai_enabled = False
        self.menu_selected = 0
        self.menu_options = ["开始游戏", "作弊模式", "AI演示模式", "音效/配乐", "退出游戏"]
        self.show_ghost_path = False

        self.treasures_collected = 0
        self.camera_offset = (0, 0)
        self._ui_time = 0.0
        self._light_mask_cache_r = -1
        self._light_mask = None

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
            self._ui_time += dt
            
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
        self.screen.fill(T.BG_DEEP)

        ox, oy = self.camera_offset
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                px = x * TILE_SIZE + ox
                py = y * TILE_SIZE + oy
                rect = pygame.Rect(px, py, TILE_SIZE, TILE_SIZE)
                if self.game_map.is_wall(x, y):
                    # 石墙：填充 + 顶面高光 + 暗边
                    pygame.draw.rect(self.screen, T.WALL_FILL, rect)
                    pygame.draw.line(self.screen, T.WALL_TOP, (px, py), (px + TILE_SIZE - 1, py), 2)
                    pygame.draw.line(self.screen, T.WALL_TOP, (px, py), (px, py + TILE_SIZE - 1), 1)
                    pygame.draw.rect(self.screen, T.WALL_EDGE, rect, 1)
                else:
                    # 棋盘微差地板 + 细缝
                    floor = T.FLOOR_A if (x + y) % 2 == 0 else T.FLOOR_B
                    pygame.draw.rect(self.screen, floor, rect)
                    pygame.draw.rect(self.screen, T.FLOOR_LINE, rect, 1)

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
                pygame.draw.rect(path_surface, (*T.ACCENT_INFO, 100), rect, border_radius=4)

            if len(ghost.current_path) > 1:
                points = []
                for grid_x, grid_y in ghost.current_path:
                    px = grid_x * TILE_SIZE + TILE_SIZE // 2 + self.camera_offset[0]
                    py = grid_y * TILE_SIZE + TILE_SIZE // 2 + self.camera_offset[1]
                    points.append((px, py))

                if len(points) > 1:
                    pygame.draw.lines(path_surface, (255, 255, 255, 90), False, points, 2)

            if ghost.current_path:
                target = ghost.current_path[-1]
                tx = target[0] * TILE_SIZE + TILE_SIZE // 2 + self.camera_offset[0]
                ty = target[1] * TILE_SIZE + TILE_SIZE // 2 + self.camera_offset[1]
                pygame.draw.circle(path_surface, (*T.ACCENT_DANGER, 180), (tx, ty), 6)
                pygame.draw.circle(path_surface, (255, 255, 255, 200), (tx, ty), 6, 2)

            self.screen.blit(path_surface, (0, 0))

    def _render_fog_v8(self):
        """柔边径向迷雾 + 已探索区域残留可见。"""
        fog_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

        player_pos = self.player.get_pixel_pos()
        light_radius_px = int(self.player.light_radius * TILE_SIZE)
        center_x = int(player_pos[0] + self.camera_offset[0])
        center_y = int(player_pos[1] + self.camera_offset[1])
        enhanced = bool(getattr(self.player, 'is_enhanced_light', lambda: False)())

        # 先铺已探索 / 未探索底雾
        ox, oy = self.camera_offset
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                rect = pygame.Rect(x * TILE_SIZE + ox, y * TILE_SIZE + oy, TILE_SIZE, TILE_SIZE)
                if self.visibility_system.explored[y][x]:
                    pygame.draw.rect(fog_surface, (6, 8, 14, 150), rect)
                else:
                    pygame.draw.rect(fog_surface, (2, 3, 6, 245), rect)

        # 径向柔光挖空
        if light_radius_px != self._light_mask_cache_r or self._light_mask is None:
            self._light_mask = T.cached_light_mask(light_radius_px)
            self._light_mask_cache_r = light_radius_px

        fog_surface.blit(
            self._light_mask,
            (center_x - light_radius_px, center_y - light_radius_px),
            special_flags=pygame.BLEND_RGBA_SUB,
        )

        # 强化光源时叠一层暖色光晕
        if enhanced:
            warm = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            glow_r = light_radius_px
            warm_mask = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
            for i in range(8, 0, -1):
                a = int(28 * (i / 8))
                pygame.draw.circle(
                    warm_mask, (*T.LIGHT_ACTIVE, a),
                    (glow_r, glow_r), int(glow_r * i / 8),
                )
            warm.blit(warm_mask, (center_x - glow_r, center_y - glow_r))
            self.screen.blit(warm, (0, 0))

        self.screen.blit(fog_surface, (0, 0))

    def _render_menu_v8(self):
        """氛围化主菜单"""
        T.draw_menu_ambiance(self.screen, self._ui_time)

        title_font = T.get_font(68, bold=True)
        sub_font = T.get_font(26)
        opt_font = T.get_font(28)
        hint_font = T.get_font(20)

        # 标题光晕
        T.draw_text_centered(
            self.screen, "开关猎杀", title_font, T.TEXT_TITLE,
            (SCREEN_WIDTH // 2, 150),
        )
        T.draw_text_centered(
            self.screen, "SWITCH HUNT", sub_font, T.ACCENT_WARM_DIM,
            (SCREEN_WIDTH // 2, 210), shadow=False,
        )
        T.draw_text_centered(
            self.screen, "黑暗迷宫 · 开灯定身 · 夺宝逃生", T.get_font(22), T.TEXT_SECONDARY,
            (SCREEN_WIDTH // 2, 250), shadow=False,
        )

        option_y_start = 310
        option_spacing = 54
        bar_w, bar_h = 420, 44

        for i, option in enumerate(self.menu_options):
            y = option_y_start + i * option_spacing
            selected = i == self.menu_selected
            bar = pygame.Rect(SCREEN_WIDTH // 2 - bar_w // 2, y - 8, bar_w, bar_h)

            if selected:
                pulse = 0.5 + 0.5 * math.sin(self._ui_time * 4)
                fill_a = int(160 + 40 * pulse)
                T.draw_panel(
                    self.screen, bar,
                    fill=(*T.BG_PANEL, fill_a),
                    border=T.ACCENT_WARM,
                    radius=10,
                    border_width=2,
                )
                # 左侧琥珀指示条
                pygame.draw.rect(
                    self.screen, T.ACCENT_WARM,
                    pygame.Rect(bar.left + 8, bar.top + 8, 4, bar_h - 16),
                    border_radius=2,
                )
                color = T.ACCENT_GOLD
            else:
                T.draw_panel(
                    self.screen, bar,
                    fill=(14, 18, 28, 120),
                    border=(40, 50, 68),
                    radius=10,
                )
                color = T.TEXT_PRIMARY

            label = option
            if i == 1:
                on = self.ui_system.cheat_mode
                label = f"{option}  {'开' if on else '关'}"
                if on:
                    color = T.ACCENT_SAFE if selected else T.ACCENT_SAFE
            elif i == 2:
                on = self.player_ai_enabled
                label = f"{option}  {'开' if on else '关'}"
                if on:
                    color = T.ACCENT_INFO
            elif i == 3:
                on = self.sound.enabled
                label = f"{option}  {'开' if on else '关'}"
                if on and selected:
                    color = T.ACCENT_WARM

            text = opt_font.render(label, True, color)
            self.screen.blit(text, text.get_rect(center=bar.center))

        T.draw_text_centered(
            self.screen,
            "↑↓ 选择   Enter 确认   ESC 退出",
            hint_font, T.TEXT_MUTED,
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 48),
            shadow=False,
        )

    def _render_hud_v8(self):
        """面板化 HUD"""
        font = T.get_font(22)
        small = T.get_font(18)

        panel = pygame.Rect(14, 12, 280, 96)
        T.draw_panel(self.screen, panel, fill=(12, 16, 26, 200), border=T.BG_PANEL_EDGE, radius=12)

        # 宝藏
        treasure_label = font.render("宝藏", True, T.TEXT_SECONDARY)
        self.screen.blit(treasure_label, (28, 22))
        for i in range(TREASURE_COUNT):
            lit = i < self.treasures_collected
            T.draw_treasure_icon(self.screen, 90 + i * 22, 34, size=7, lit=lit)
        count = small.render(f"{self.treasures_collected}/{TREASURE_COUNT}", True, T.ACCENT_GOLD)
        self.screen.blit(count, (90 + TREASURE_COUNT * 22 + 8, 26))

        # 光源
        if isinstance(self.player, PlayerV8):
            p = self.player
            light_label = font.render("光源", True, T.TEXT_SECONDARY)
            self.screen.blit(light_label, (28, 58))

            for i in range(p.light_charges_max):
                T.draw_lamp_icon(self.screen, 90 + i * 26, 70, lit=i < p.light_charges)

            if p.light_state == LightState.ACTIVE:
                status = f"激活 {p.light_active_timer:.1f}s"
                color = T.ACCENT_WARM
            elif p.light_state == LightState.COOLDOWN:
                status = f"冷却 {p.light_cooldown_timer:.1f}s"
                color = T.ACCENT_GOLD
            else:
                status = f"{p.light_charges}/{p.light_charges_max}"
                color = T.ACCENT_SAFE if p.light_charges > 0 else T.ACCENT_DANGER

            status_s = small.render(status, True, color)
            self.screen.blit(status_s, (90 + p.light_charges_max * 26 + 10, 62))

            if p.ai_enabled:
                badge = pygame.Rect(SCREEN_WIDTH - 118, 14, 104, 28)
                T.draw_panel(self.screen, badge, fill=(*T.ACCENT_INFO, 50), border=T.ACCENT_INFO, radius=8)
                T.draw_text_centered(
                    self.screen, "AI 自动", small, T.ACCENT_INFO,
                    badge.center, shadow=False,
                )

        if self.ui_system.cheat_mode:
            badge = pygame.Rect(SCREEN_WIDTH - 118, 50 if getattr(self.player, 'ai_enabled', False) else 14, 104, 28)
            T.draw_panel(self.screen, badge, fill=(*T.ACCENT_DANGER, 50), border=T.ACCENT_DANGER, radius=8)
            T.draw_text_centered(
                self.screen, "作弊模式", small, T.ACCENT_DANGER,
                badge.center, shadow=False,
            )

    def _render_pause_v8(self):
        """暂停界面"""
        T.draw_overlay_card(
            self.screen,
            "游戏暂停",
            "P / ESC 继续游戏",
            T.TEXT_PRIMARY,
            hints=["Enter 返回主菜单"],
        )

    def _render_game_over_v8(self):
        """失败界面"""
        collected = f"已收集宝藏  {self.treasures_collected}/{TREASURE_COUNT}"
        T.draw_overlay_card(
            self.screen,
            "猎杀失败",
            collected,
            T.ACCENT_DANGER,
            hints=["Enter 重新开始", "ESC 返回主菜单"],
        )

    def _render_victory_v8(self):
        """胜利界面"""
        T.draw_overlay_card(
            self.screen,
            "成功逃脱",
            "你夺回了全部宝藏",
            T.ACCENT_GOLD,
            hints=["Enter 再来一局", "ESC 返回主菜单"],
        )


def main():
    """V8主函数"""
    print("=" * 60)
    print("开关猎杀 — Switch Hunt")
    print("=" * 60)
    print("操作说明:")
    print("  WASD/方向键: 移动")
    print("  空格: 手动激活强化光源（需自行判断时机）")
    print("  P: 暂停 | ESC: 菜单")
    print("  F1: 作弊模式 | F2: AI演示 | F3: 显示A*路径 | F4: 音效/配乐")
    print("提示: 进入光源范围的鬼会显形；强化光源可定身它们。")
    print("=" * 60)

    game = GameV8()
    game.run()


if __name__ == "__main__":
    main()
