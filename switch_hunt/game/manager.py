"""V7 GameManager：基础主循环与状态机。"""
from __future__ import annotations

from typing import List, Optional

import pygame
import random

from switch_hunt.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, TILE_SIZE, MAP_WIDTH, MAP_HEIGHT,
    TREASURE_COUNT, PLAYER_SPEED, TREASURE_ENERGY_RESTORE,
    GHOST_FREEZE_DURATION, LIGHT_RADIUS_ENHANCED,
    COLOR_BLACK, COLOR_WHITE,
)
from switch_hunt.enums import GameState, LightMode, GhostState
from switch_hunt.utils import grid_to_pixel, pixel_to_grid, distance
from switch_hunt.core.map import Map
from switch_hunt.core.pathfinding import AStarPathfinder
from switch_hunt.core.entities.player import Player
from switch_hunt.core.entities.ghost import Ghost
from switch_hunt.core.entities.treasure import Treasure
from switch_hunt.core.visibility import VisibilitySystem
from switch_hunt.game.ui_base import UISystem

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
                from switch_hunt.rl.agent import DQNAI
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
                from switch_hunt.rl.agent import DQNAI
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
