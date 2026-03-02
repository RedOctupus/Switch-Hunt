"""
DQN训练环境 v8.0 - 网格对齐方向学习系统
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['DQN_TRAINING'] = '1'

import random
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json  # V8.22: 移到顶部
from datetime import datetime
from collections import deque

from switch_hunt_v8_game import GameV8, DQNGhostV8, PlayerV8, GhostState, LightState
from switch_hunt_v7_base import LightMode
from switch_hunt_v7_base import TILE_SIZE
from maze_system import Map
from config_v8 import GHOST_REWARD, LIGHT_SYSTEM, MAX_EPISODE_STEPS, CHECKPOINT_CONFIG
from config_v8 import MAX_DISCRETE_STEPS, CURRICULUM, GHOST_MOVE
from dqn_model_v8 import ConfigurableDQNAI


class CheckpointManager:
    """V8.22: 训练检查点管理器
    
    职责:
    - 定期保存模型检查点
    - 管理检查点文件（保留最近N个）
    - 保存训练统计信息
    
    Clean Code原则:
    - 单一职责: 只处理检查点相关逻辑
    - 开闭原则: 通过配置控制行为
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: 检查点配置字典，默认使用CHECKPOINT_CONFIG
        """
        self.config = config or CHECKPOINT_CONFIG
        self.enabled = self.config.get('enabled', True)
        self.interval = self.config.get('interval', 10)
        self.max_keep = self.config.get('max_keep', 5)
        self.save_dir = self.config.get('save_dir', 'checkpoints')
        self.save_stats = self.config.get('save_stats', True)
        
        # 记录已保存的检查点
        self.saved_checkpoints = []
        
        if self.enabled:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"[V8.22] 检查点管理器已启用: 每{self.interval}轮保存，保留最近{self.max_keep}个")
    
    def should_save(self, episode):
        """检查是否应该在本轮保存检查点
        
        Args:
            episode: 当前轮数
            
        Returns:
            bool: 是否应该保存
        """
        if not self.enabled:
            return False
        return episode % self.interval == 0
    
    def save_checkpoint(self, agent, episode, stats=None):
        """保存检查点
        
        Args:
            agent: DQN代理
            episode: 当前轮数
            stats: 训练统计字典（可选）
            
        Returns:
            str: 保存的文件路径，失败返回None
        """
        if not self.enabled:
            return None
        
        # 生成文件名: checkpoint_ep{轮数}_{时间戳}.pth
        timestamp = datetime.now().strftime('%m%d_%H%M%S')
        filename = f'checkpoint_ep{episode:04d}_{timestamp}.pth'
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            # 保存模型
            agent.save(filepath)
            
            # 记录检查点
            self.saved_checkpoints.append({
                'episode': episode,
                'filepath': filepath,
                'timestamp': timestamp
            })
            
            print(f"[V8.22] 检查点已保存: {filename}")
            
            # 保存训练统计
            if self.save_stats and stats is not None:
                self._save_stats(episode, stats, timestamp)
            
            # 清理旧检查点
            self._cleanup_old_checkpoints()
            
            return filepath
            
        except Exception as e:
            print(f"[V8.22 Warning] 保存检查点失败: {e}")
            return None
    
    def _save_stats(self, episode, stats, timestamp):
        """保存训练统计信息
        
        V8.22改进: 保存时记录stats文件路径，避免后续通过字符串替换查找
        """
        stats_filename = f'stats_ep{episode:04d}_{timestamp}.json'
        stats_filepath = os.path.join(self.save_dir, stats_filename)
        
        try:
            with open(stats_filepath, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            # V8.22改进: 记录stats文件路径到最新的检查点记录
            if self.saved_checkpoints:
                self.saved_checkpoints[-1]['stats_filepath'] = stats_filepath
                
        except Exception as e:
            print(f"[V8.22 Warning] 保存统计信息失败: {e}")
    
    def load_checkpoint(self, agent, checkpoint_path=None):
        """加载检查点
        
        V8.22新增: 支持从指定路径或最新检查点恢复
        
        Args:
            agent: DQN代理
            checkpoint_path: 指定检查点路径，None表示加载最新
            
        Returns:
            dict: 检查点信息，失败返回None
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print(f"[V8.22 Warning] 检查点不存在: {checkpoint_path}")
            return None
        
        try:
            agent.load(checkpoint_path)
            
            # 找到对应的检查点记录
            checkpoint_info = None
            for cp in self.saved_checkpoints:
                if cp['filepath'] == checkpoint_path:
                    checkpoint_info = cp
                    break
            
            print(f"[V8.22] 检查点已加载: {os.path.basename(checkpoint_path)}")
            return checkpoint_info
            
        except Exception as e:
            print(f"[V8.22 Warning] 加载检查点失败: {e}")
            return None
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点，只保留最近的N个
        
        V8.22改进: 使用记录的stats_filepath而不是字符串替换
        """
        if self.max_keep <= 0:
            return  # 0表示不限制
        
        while len(self.saved_checkpoints) > self.max_keep:
            old_checkpoint = self.saved_checkpoints.pop(0)
            old_filepath = old_checkpoint['filepath']
            
            try:
                # 删除模型文件
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)
                    print(f"[V8.22] 清理旧检查点: {os.path.basename(old_filepath)}")
                
                # V8.22改进: 使用记录的stats_filepath删除统计文件
                old_stats = old_checkpoint.get('stats_filepath')
                if old_stats and os.path.exists(old_stats):
                    os.remove(old_stats)
                    
            except Exception as e:
                print(f"[V8.22 Warning] 清理旧检查点失败: {e}")
    
    def get_latest_checkpoint(self):
        """获取最新的检查点路径
        
        Returns:
            str: 最新检查点路径，不存在返回None
        """
        if not self.saved_checkpoints:
            return None
        return self.saved_checkpoints[-1]['filepath']
    
    def list_checkpoints(self):
        """列出所有检查点
        
        Returns:
            list: 检查点信息列表
        """
        return self.saved_checkpoints.copy()


class SwitchHuntTrainingEnvV8:
    """V8: 网格对齐方向学习训练环境"""
    
    def __init__(self, render=False, opponent_type='simple'):
        self.render_enabled = render
        
        pygame.init()
        if render:
            self.screen = pygame.display.set_mode((1024, 768))
            pygame.display.set_caption("DQN Training v8.0 - Grid-Aligned Direction Learning")
        else:
            self.screen = pygame.Surface((1024, 768))
        
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        """V8.25: 重置环境 — 鬼出生在距离玩家适当A*步数的位置（GHOST_SPAWN配置）

        修改原因:
        - V8.19: target_steps=5，鬼太近（5步即抓到），玩家容易轻易引诱
        - V8.25: 改为随机范围 [min_steps, max_steps]，确保距离合理
        """
        self.game = GameV8()
        self.game.screen = self.screen

        # 先正常初始化（init_game 已含 V8.25 的游戏模式出生距离检查）
        self.game.init_game()

        # V8.25: 训练模式额外调整鬼位置到合适A*步数范围
        self._adjust_ghost_to_path_distance()

        # 初始化静止玩家光源AI状态（Phase 2使用）
        self._static_light_reacting = False
        self._static_light_timer = 0.0
        
        if len(self.game.ghosts) > 1:
            self.game.ghosts = [self.game.ghosts[0]]
        self.ghost = self.game.ghosts[0]
        
        # 加载模型
        try:
            from dqn_model_v8 import DQNAI
            model_paths = ['models/ghost_v8.pth', '../models/ghost_v8.pth', '../../models/ghost_v8.pth']
            for model_path in model_paths:
                if os.path.exists(model_path):
                    self.ghost.dqn_ai = DQNAI(state_channels=7, epsilon=0.0)
                    self.ghost.dqn_ai.load(model_path)
                    self.ghost.use_dqn = True
                    if os.environ.get('DQN_TRAINING') != '1':
                        print(f"[V8] Ghost model loaded from {model_path}")
                    break
        except Exception as e:
            if os.environ.get('DQN_TRAINING') != '1':
                print(f"[V8] No model found: {e}")
        
        # V8: 初始化A*路径和方向
        self.ghost.update_path()
        self.planned_direction = self.ghost.planned_direction
        
        # V8更新: 配置玩家（静止或AI控制）
        # 设置光源次数（从LIGHT_SYSTEM配置）
        self.game.player.light_charges = LIGHT_SYSTEM['initial_charges']
        self.game.player.light_charges_max = LIGHT_SYSTEM['max_charges']
        
        # V8.26: 玩家AI状态由训练循环中的课程学习覆盖，reset只设置默认静止状态
        # [V8.26] 注意：Phase3已禁用，所有回合 ai_enabled=False（静止玩家）
        self.game.player.ai_enabled = False
        self.game.player._update_ai_path()  # 预初始化路径以防Phase3立即启用AI
        
        self.episode_step = 0
        
        return self._get_ghost_state(), None
    
    def _adjust_ghost_to_path_distance(self):
        """V8.25: 将鬼调整到距离玩家 [min_steps, max_steps] A*步数内的随机位置。

        规则:
        - min_steps~max_steps来自GHOST_SPAWN配置（默认7~12步）
        - 随机选取步数，避免鬼总在固定距离出现
        - 若路径太短则尽量放到最远可行位置
        """
        from config_v8 import GHOST_SPAWN
        from switch_hunt_v8_game import grid_to_pixel

        if not self.game.ghosts or not self.game.player:
            return

        ghost = self.game.ghosts[0]
        player = self.game.player

        ghost_grid = ghost.grid_pos
        player_grid = player.get_grid_pos()

        path = ghost.pathfinder.find_path(ghost_grid, player_grid)
        if not path or len(path) < 2:
            return

        path_steps = len(path) - 1  # 不含起点的步数
        min_s = GHOST_SPAWN.get('min_steps', 7)
        max_s = GHOST_SPAWN.get('max_steps', 12)

        if path_steps < min_s:
            # 路径太短（鬼已经很近），保持原位或放到路径尽头
            return

        # 随机选择 [min_s, min(max_s, path_steps)] 中的目标步数
        target_steps = random.randint(min_s, min(max_s, path_steps))

        # 路径索引：path[0]=鬼当前位置，path[-1]=玩家位置
        # 距离玩家 target_steps 步 → 路径中倒数第 target_steps+1 个元素
        target_index = path_steps - target_steps
        new_ghost_pos = path[target_index]

        old_pos = ghost.grid_pos
        ghost.grid_pos = new_ghost_pos
        ghost.pos = list(grid_to_pixel(new_ghost_pos[0], new_ghost_pos[1]))
        ghost.target_grid = None
        ghost.is_moving = False
        ghost.move_progress = 0.0
        ghost.current_path = path[target_index:]

        print(f"[V8.25] 鬼位置调整: {old_pos} → {new_ghost_pos}（距玩家{target_steps}步，范围{min_s}~{max_s}）")
    
    def _get_ghost_state(self):
        """获取鬼的状态编码"""
        return self.ghost.get_state()
    
    def step_train_ghost(self, action):
        """V8.21: 训练步骤 - 修复方向奖励同步问题
        
        核心修复:
        1. 每步更新A*路径和planned_direction（原来每5步，导致方向不同步）
        2. 在动作执行前就确定方向奖励基准（避免动作与评判标准不一致）
        
        Args:
            action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        
        Returns:
            next_state, reward, done, info
        """
        dt = 1/60
        
        # V8.21修复: 在动作执行前更新路径和方向（确保评判标准与当前状态一致）
        self.ghost.update_path()
        self.planned_direction = self.ghost.planned_direction
        
        # 记录执行前状态
        was_stunned = self.ghost.state == GhostState.STUNNED
        
        # V8.21: 在动作执行前确定方向奖励基准
        # 这样即使路径在移动过程中更新，我们仍按执行前的标准评判
        direction_reference = self.planned_direction
        
        # V8: 执行动作
        moved = False
        if self.ghost.state != GhostState.STUNNED:
            # 如果不在移动中，开始新移动
            if not self.ghost.is_moving:
                moved = self.ghost.apply_action(action, dt)
            else:
                # 继续当前移动
                moved = self.ghost._continue_move(dt)
        
        # V8: 更新鬼状态（包括定身计时器递减）
        self.ghost.update_for_dqn_training(dt, self.game.player)
        
        # V8.21: 移除原来的5步路径更新（已在开头更新）
        # 保持path_update_counter用于调试统计
        self.ghost.path_update_counter += 1
        if self.ghost.path_update_counter >= 5:
            self.ghost.path_update_counter = 0
        
        # V8更新: 玩家更新
        if self.game.player.ai_enabled:
            # AI控制：自动寻路
            self.game.player.update(dt)
        else:
            # 玩家静止：手动更新光源状态，但不移动
            self.game.player._update_light_mode(dt)
            # 确保玩家不移动
            self.game.player.keys_pressed = {'up': False, 'down': False, 'left': False, 'right': False, 'enhance': False}
        
        # 检查玩家是否开启光源（AI控制或玩家按键）
        self._check_player_light_activation()
        # 检查宝藏收集
        self.game._check_treasures()
        # V8更新: 训练模式下不重置光源（固定次数）
        
        # 检查光源定身
        self._check_light_stun(dt)
        
        # 检查碰撞（定身的鬼不能抓玩家）
        if self.ghost.state == GhostState.STUNNED:
            touched_player = False
        else:
            touched_player = self.ghost.check_collision(self.game.player)
        
        # V8.17: 计算奖励（简化版方向趋势奖励）
        # 检查玩家是否开启光源
        player_light_active = (hasattr(self.game.player, 'light_state') and 
                               self.game.player.light_state == LightState.ACTIVE)
        
        # V8.21: 使用动作执行前的方向基准（direction_reference）而非实时更新后的
        reward = self._calculate_direction_reward(
            action, direction_reference, moved, was_stunned, touched_player,
            player_light_active
        )
        
        # 结束条件
        done = touched_player or self.episode_step > MAX_EPISODE_STEPS
        next_state = self._get_ghost_state()
        
        self.episode_step += 1
        
        # V8.17: 简化调试信息
        info = {
            'step': self.episode_step,
            'caught': touched_player,
            'stunned': self.ghost.state == GhostState.STUNNED,
            'moved': moved,
            'planned_direction': self.planned_direction,
            'actual_direction': action,
            'is_moving': self.ghost.is_moving,
            # V8.17调试信息
            'player_light_active': player_light_active,
            'reward': reward,
        }
        
        return next_state, reward, done, info
    
    def _calculate_direction_reward(self, actual_direction, planned_direction, 
                                     moved, was_stunned, touched_player,
                                     player_light_active=False):
        """V8.21: 修复版方向趋势奖励计算
        
        关键修复:
        1. 移除"必须moved才给方向奖励"的限制（网格移动中moved很稀疏）
        2. 只要planned_direction存在，就根据动作方向给奖励/惩罚
        3. 增加调试日志（每100步打印一次）
        
        核心逻辑:
        - 抓到玩家: +2000分
        - 实际方向 == A*路径方向: +2分
        - 实际方向 != A*路径方向: -3分
        - 撞墙(未移动): -0.5分
        - 玩家开灯时鬼没被定身: +500分
        - 被定身: -300分
        """
        reward = 0.0
        direction_reward = 0.0
        
        # 抓住玩家 (+2000分)
        if touched_player:
            return GHOST_REWARD.get('catch_player', 2000.0)
        
        # 定身惩罚 (-300分)
        if self.ghost.state == GhostState.STUNNED and not was_stunned:
            reward += GHOST_REWARD.get('stunned_penalty', -300.0)
        
        # V8.17: 玩家开灯时鬼没被定身 (+500分)
        if player_light_active and self.ghost.state != GhostState.STUNNED:
            reward += GHOST_REWARD.get('evade_light_bonus', 500.0)
        
        # V8.21修复: 路径方向奖励/惩罚（移除"必须moved"的限制）
        if planned_direction is not None:
            if actual_direction == planned_direction:
                direction_reward = GHOST_REWARD.get('correct_direction', 2.0)  # +2
            else:
                direction_reward = GHOST_REWARD.get('wrong_direction', -3.0)  # -3
            
            reward += direction_reward
            
            # V8.21: 调试日志（每100步打印一次方向奖励情况）
            if self.episode_step % 100 == 0 and os.environ.get('DQN_TRAINING') == '1':
                dir_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                planned_name = dir_names[planned_direction] if planned_direction is not None else 'None'
                actual_name = dir_names[actual_direction] if actual_direction is not None else 'None'
                print(f"[V8.21 Debug] Step {self.episode_step}: 计划={planned_name}, 实际={actual_name}, "
                      f"方向奖励={direction_reward:+.1f}, 移动={'成功' if moved else '失败/进行中'}")
        
        # 撞墙惩罚（只有当确实尝试移动但失败时）
        if not moved and planned_direction is not None:
            # 检查是否是因为撞墙
            if not self.ghost.is_moving:
                reward += GHOST_REWARD.get('wall_hit', -0.5)
        
        return reward
    
    def _check_light_stun(self, dt=1/60):
        """V8.25: 定身检测 — 与游戏模式统一规则。
        鬼进入 enhanced_radius（3格）后持续曝光 stun_exposure_time（1秒）才定身。
        此机制给 DQN 留出逃脱窗口，同时与游戏体验一致。
        """
        player = self.game.player
        ghost = self.ghost

        if hasattr(player, 'light_state') and player.light_state == LightState.ACTIVE:
            if ghost.state == GhostState.STUNNED:
                ghost._stun_exposure = 0.0
                return
            stun_r_px = LIGHT_SYSTEM.get('enhanced_radius', 3) * TILE_SIZE
            exposure_needed = LIGHT_SYSTEM.get('stun_exposure_time', 1.0)
            player_pos = player.get_pixel_pos()
            dist = ((ghost.pos[0] - player_pos[0])**2 +
                    (ghost.pos[1] - player_pos[1])**2) ** 0.5
            if dist <= stun_r_px:
                ghost._stun_exposure = getattr(ghost, '_stun_exposure', 0.0) + dt
                if ghost._stun_exposure >= exposure_needed:
                    ghost._stun_exposure = 0.0
                    ghost.freeze(LIGHT_SYSTEM['stun_duration'] / 1000.0)
            else:
                ghost._stun_exposure = 0.0
        else:
            self.ghost._stun_exposure = 0.0
    
    def _update_static_player_light(self, dt):
        """V8.25: Phase 2 静止玩家的光源AI（含300ms反应延迟）。
        玩家不移动，但检测到鬼后延迟 light_reaction_delay 秒开灯。
        """
        player = self.game.player
        if player.light_state != LightState.IDLE or player.light_charges <= 0:
            self._static_light_reacting = False
            self._static_light_timer = 0.0
            return

        ghost_in_range = False
        for g in self.game.ghosts:
            if g.state == GhostState.STUNNED:
                continue
            dist = ((player.pos[0] - g.pos[0])**2 +
                    (player.pos[1] - g.pos[1])**2) ** 0.5
            if dist / TILE_SIZE <= LIGHT_SYSTEM.get('auto_light_range', 3):
                ghost_in_range = True
                break

        if ghost_in_range:
            if not self._static_light_reacting:
                self._static_light_reacting = True
                self._static_light_timer = LIGHT_SYSTEM.get('light_reaction_delay', 0.3)
            else:
                self._static_light_timer -= dt
                if self._static_light_timer <= 0:
                    self._static_light_reacting = False
                    player.keys_pressed['enhance'] = True
        else:
            self._static_light_reacting = False
            self._static_light_timer = 0.0

    def _check_player_light_activation(self):
        """检查玩家是否激活光源（处理AI按键输入）"""
        player = self.game.player
        if not hasattr(player, 'light_state'):
            return
        
        # 检查玩家是否按下开灯键（AI可能设置了这个）
        if player.keys_pressed.get('enhance', False):
            if (player.light_state == LightState.IDLE and 
                player.light_charges > 0):
                # 激活光源
                player.light_charges -= 1
                player.light_state = LightState.ACTIVE
                player.light_active_timer = LIGHT_SYSTEM['active_duration']
                player.light_mode = LightMode.ENHANCED
                player.light_radius = LIGHT_SYSTEM.get('enhanced_radius', 3)
                # 玩家开灯（静默）
                pass
    
    def render(self, last_action=None, last_reward=None):
        """渲染游戏画面，包含A*路径调试信息"""
        if not self.render_enabled:
            return
        
        self.screen.fill((0, 0, 0))
        
        # 训练时开启作弊模式（显示全地图）
        if hasattr(self.game, 'ui_system'):
            self.game.ui_system.cheat_mode = True
        
        # 使用游戏渲染
        if hasattr(self.game, '_render_v8'):
            self.game._render_v8()
        
        # 渲染A*路径（如果存在）
        if hasattr(self.ghost, 'current_path') and self.ghost.current_path:
            self._render_astar_path()
        
        # 渲染训练信息面板（半透明背景）
        panel_surface = pygame.Surface((300, 200), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 180))
        self.screen.blit(panel_surface, (5, 5))
        
        # 渲染训练信息
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 20)
        
        # V8.17: A*路径从鬼当前位置计算，鬼永远在路径上
        path_status = "ON PATH"
        path_color = (0, 255, 0)
        
        # 检查方向是否正确（如果有上一步动作）
        dir_correct = None
        if last_action is not None and self.planned_direction is not None:
            dir_correct = (last_action == self.planned_direction)
        
        info_texts = [
            (f"Step: {self.episode_step}", (255, 255, 255)),
            (f"Ghost: {self.ghost.grid_pos} -> Player: {self.game.player.get_grid_pos()}", (255, 255, 255)),
            (f"A* Path: {path_status}", path_color),
            (f"Planned: {self._dir_to_str(self.planned_direction)} | Actual: {self._dir_to_str(last_action)}", (255, 255, 255)),
        ]
        
        # 添加方向正确性
        if dir_correct is not None:
            dir_text = "CORRECT" if dir_correct else "WRONG"
            dir_color = (0, 255, 0) if dir_correct else (255, 0, 0)
            info_texts.append((f"Direction: {dir_text}", dir_color))
        
        # 添加奖励信息
        if last_reward is not None:
            reward_color = (0, 255, 0) if last_reward > 0 else (255, 0, 0) if last_reward < 0 else (255, 255, 255)
            info_texts.append((f"Reward: {last_reward:+.2f}", reward_color))
        
        # 添加移动状态
        move_text = "Moving" if self.ghost.is_moving else "Idle"
        info_texts.append((f"State: {move_text}", (255, 255, 255)))
        
        y_offset = 10
        for text, color in info_texts:
            surface = font.render(text, True, color)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 22
        
        # 添加图例说明
        legend_y = y_offset + 5
        pygame.draw.line(self.screen, (0, 255, 255), (10, legend_y + 10), (30, legend_y + 10), 3)
        self.screen.blit(small_font.render("A* Path", True, (200, 200, 200)), (35, legend_y + 5))
        
        pygame.draw.circle(self.screen, (255, 0, 0), (100, legend_y + 10), 5)
        self.screen.blit(small_font.render("Target", True, (200, 200, 200)), (110, legend_y + 5))
        
        pygame.draw.circle(self.screen, (0, 255, 0), (180, legend_y + 10), 5)
        self.screen.blit(small_font.render("On Path", True, (200, 200, 200)), (190, legend_y + 5))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def _render_astar_path(self):
        """渲染鬼的A*路径（半透明覆盖层）"""
        if not self.ghost.current_path:
            return
        
        path_surface = pygame.Surface((1024, 768), pygame.SRCALPHA)
        
        # 绘制路径格子（青色半透明）
        for grid_x, grid_y in self.ghost.current_path:
            rect = pygame.Rect(
                grid_x * 32 + self.game.camera_offset[0] + 8,
                grid_y * 32 + self.game.camera_offset[1] + 8,
                16, 16
            )
            pygame.draw.rect(path_surface, (0, 255, 255, 100), rect, border_radius=3)
        
        # 绘制路径连线（白色半透明）
        if len(self.ghost.current_path) > 1:
            points = []
            for grid_x, grid_y in self.ghost.current_path:
                px = grid_x * 32 + 16 + self.game.camera_offset[0]
                py = grid_y * 32 + 16 + self.game.camera_offset[1]
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(path_surface, (255, 255, 255, 150), False, points, 2)
        
        # 绘制目标点（红色）
        if self.ghost.current_path:
            target = self.ghost.current_path[-1]
            tx = target[0] * 32 + 16 + self.game.camera_offset[0]
            ty = target[1] * 32 + 16 + self.game.camera_offset[1]
            pygame.draw.circle(path_surface, (255, 0, 0, 200), (tx, ty), 8)
            pygame.draw.circle(path_surface, (255, 255, 255, 255), (tx, ty), 8, 2)
        
        self.screen.blit(path_surface, (0, 0))
    
    # V8.17: 移除 _is_ghost_on_astar_path() 方法
    # 原因: A*路径从鬼当前位置计算，鬼永远在路径上，无需检查
    
    # V8.17: 移除 _is_ghost_in_light_range() 方法  
    # 原因: 训练时只有鬼接近玩家才会开灯，简化判定逻辑
    
    def _dir_to_str(self, direction):
        """方向转字符串"""
        if direction is None:
            return "-"
        return ["UP", "DOWN", "LEFT", "RIGHT"][direction]
    
    # ================================================================
    # V8.23: 离散化训练接口（替代 step_train_ghost）
    # ================================================================

    def _get_grid_distance(self):
        """获取鬼与玩家的曼哈顿距离（格子单位）"""
        gx, gy = self.ghost.grid_pos
        px, py = self.game.player.get_grid_pos()
        return abs(gx - px) + abs(gy - py)

    def _detect_phase(self):
        """检测当前博弈阶段
        Returns: 'light_active' | 'sprint' | 'stalk'
        """
        player = self.game.player
        light_state = getattr(player, 'light_state', None)
        charges = getattr(player, 'light_charges', 0)
        if light_state == LightState.ACTIVE:
            return 'light_active'
        elif charges == 0:
            return 'sprint'
        else:
            return 'stalk'

    def _calculate_phase_reward(self, action, planned_direction, phase,
                                 was_stunned, touched_player, bait_triggered,
                                 prev_dist, curr_dist):
        """V8.23: 三阶段博弈奖励（含距离变化密集信号）"""
        if touched_player:
            return GHOST_REWARD['catch_player']

        reward = 0.0
        dist_delta = prev_dist - curr_dist  # 正=靠近，负=远离

        if phase == 'stalk':
            # 方向奖励（弱）
            if planned_direction is not None:
                if action == planned_direction:
                    reward += GHOST_REWARD['correct_direction_stalk']
                else:
                    reward += GHOST_REWARD['wrong_direction_stalk']
            # 距离变化奖励（连续信号）
            if dist_delta > 0:
                reward += GHOST_REWARD['approach_bonus_stalk']
            elif dist_delta < 0:
                reward += GHOST_REWARD['retreat_penalty_stalk']
            # 区域奖励
            if curr_dist <= 3:
                reward += GHOST_REWARD['in_trigger_zone']
            if curr_dist <= 2:
                reward += GHOST_REWARD['ambush_bonus']  # 冒进高风险区

        elif phase == 'sprint':
            # 方向奖励（强，全力追击）
            if planned_direction is not None:
                if action == planned_direction:
                    reward += GHOST_REWARD['correct_direction_sprint']
                else:
                    reward += GHOST_REWARD['wrong_direction_sprint']
            if dist_delta > 0:
                reward += GHOST_REWARD['approach_bonus_sprint']

        elif phase == 'light_active':
            # 定身惩罚
            if self.ghost.state == GhostState.STUNNED and not was_stunned:
                reward += GHOST_REWARD['stunned_penalty']
            # 距离变化：撤退有益，靠近有惩罚
            if dist_delta < 0:
                reward += GHOST_REWARD['retreat_bonus_light']
            elif dist_delta > 0:
                reward += GHOST_REWARD['approach_penalty_light']

        # 引诱成功（玩家开灯 + 鬼未被定身）
        if bait_triggered and self.ghost.state != GhostState.STUNNED:
            reward += GHOST_REWARD['bait_success']

        # 撞墙惩罚
        if not self.ghost.is_moving and not was_stunned:
            reward += GHOST_REWARD['wall_hit']

        return reward

    def step_discrete(self, action):
        """V8.23: 离散化训练步骤 — 1次调用 = 1次完整格间移动

        核心改进（修复Bug1）:
        原step_train_ghost每帧调用，15帧中14帧忽略action但仍计算奖励。
        现在每次完整执行一次格间移动，确保 action → 实际执行 → 奖励 的因果链正确。

        Returns: next_state, reward, done, info
        """
        dt = 1 / 60
        MAX_SUBSTEPS = 30  # 安全帧数上限（速度3.5时理论需~17帧）

        # 1. 记录执行前状态
        self.ghost.update_path()
        planned_direction = self.ghost.planned_direction
        was_stunned = (self.ghost.state == GhostState.STUNNED)
        prev_light_state = self.game.player.light_state
        prev_dist = self._get_grid_distance()
        phase = self._detect_phase()

        # V8.25: 删除sprint_mode，速度恒为玩家1.2倍，sprint仅为奖励结构标签

        # 2. 执行动作（仅此处执行，离散化的核心）
        if not was_stunned:
            self.ghost.apply_action(action, dt)

        # 3. 推进物理直到格间移动完成
        substeps = 0
        while self.ghost.is_moving and substeps < MAX_SUBSTEPS:
            self.ghost._continue_move(dt)
            self.game.player._update_light_mode(dt)
            self._check_light_stun(dt)
            substeps += 1

        # 4. 更新鬼定身计时器
        # V8.23修复: substeps=0（鬼被定身/撞墙未移动）时，按完整格间移动时间推进。
        # 否则每次定身步骤只计1帧(1/60s)，导致2s定身需120个离散步骤而非应有的~7步。
        nominal_move_time = 1.0 / GHOST_MOVE.get('speed', 5.625)  # V8.25: speed=5.625 → ≈0.178s/格
        total_dt = nominal_move_time if substeps == 0 else dt * (substeps + 1)
        if self.ghost.state == GhostState.STUNNED:
            self.ghost.stun_timer -= total_dt
            if self.ghost.stun_timer <= 0:
                self.ghost.state = GhostState.NORMAL

        # 5. 玩家移动（[V8.26] 所有阶段 ai_enabled=False，均为静止玩家+光源AI）
        if self.game.player.ai_enabled:
            for _ in range(substeps + 1):
                self.game.player.update(dt)
        else:
            # 静止玩家：不移动，但光源AI仍有300ms延迟开灯（Phase 2 引诱机制）
            self.game.player.keys_pressed = {
                'up': False, 'down': False, 'left': False,
                'right': False, 'enhance': False
            }
            # V8.25: 静止玩家光源AI（Phase 2 引诱学习核心）
            # substeps>0时光源状态机已在substep循环内更新，不重复调用以避免双重计时
            dt_step = nominal_move_time if substeps == 0 else dt * (substeps + 1)
            if self.game.player.light_charges > 0:
                self._update_static_player_light(dt_step)
            # 仅在substeps==0（鬼被定身/撞墙）时需要主动推进光源状态机
            if substeps == 0:
                self.game.player._update_light_mode(total_dt)

        self._check_player_light_activation()
        self.game._check_treasures()

        # 6. 碰撞检测
        if self.ghost.state == GhostState.STUNNED:
            touched_player = False
        else:
            touched_player = self.ghost.check_collision(self.game.player)

        # 7. 引诱检测（光源状态转变 IDLE→ACTIVE）
        curr_light_state = self.game.player.light_state
        bait_triggered = (prev_light_state == LightState.IDLE and
                          curr_light_state == LightState.ACTIVE)

        # 8. 奖励计算（每次格间决策只计算一次）
        curr_dist = self._get_grid_distance()
        reward = self._calculate_phase_reward(
            action, planned_direction, phase,
            was_stunned, touched_player, bait_triggered,
            prev_dist, curr_dist
        )

        done = touched_player or self.episode_step >= MAX_DISCRETE_STEPS
        next_state = self._get_ghost_state()
        self.episode_step += 1

        info = {
            'step': self.episode_step,
            'caught': touched_player,
            'planned_direction': planned_direction,
            'actual_direction': action,
            'phase': phase,
            'bait_triggered': bait_triggered,
            'is_moving': self.ghost.is_moving,
            'reward': reward,
        }
        return next_state, reward, done, info

    def close(self):
        """关闭环境"""
        pygame.quit()


def train_ghost_v8(episodes=500, print_every=10, render=False):
    """V8.26: 训练鬼AI - 两阶段博弈行为学习（Phase3已暂时禁用）

    用法（推荐使用 train.py 入口）:
        python train.py                        # 默认500轮，无渲染
        python train.py -e 1000 -r 1           # 1000轮，带实时渲染
        python dqn_training_system_v8.py       # 直接运行，默认500轮带渲染

    Args:
        episodes: 训练回合总数（默认500，对应两阶段课程）
        print_every: 打印频率（每N轮）
        render: 是否实时渲染
    """
    print("=" * 70)
    print("DQN Training v8.26 - Two-Phase Ghost Behavior Learning")
    print("=" * 70)
    print(f"\n训练配置: episodes={episodes}, print_every={print_every}, render={render}")
    print("\nV8.26核心特性:")
    print("  1. 离散化训练步骤（1决策=1格间移动）")
    print("  2. [V8.26] 两阶段课程：Phase1纯冲刺→Phase2引诱（静止玩家+300ms开灯）")
    print("     [注意] Phase3（AI玩家移动）已暂时禁用：玩家AI太傻导致鬼学会静止不动")
    print("  3. 鬼速度恒为玩家1.2倍（5.625格/秒）")
    print("  4. 定身规则统一：强化光源3格范围内持续1秒才定身")
    print("  5. 鬼出生距离：距玩家7~12 A*步")
    print("  6. 2格ambush_radius：高风险引诱区奖励")
    print(f"\n课程阶段: Phase1 EP1-{CURRICULUM['phase1_end']} | "
          f"Phase2a EP{CURRICULUM['phase1_end']+1}-{CURRICULUM['phase2_end']} | "
          f"Phase2b EP{CURRICULUM['phase2_end']+1}-{CURRICULUM['phase3_end']}")
    print("=" * 70)

    # 从配置文件导入所有参数
    from config_v8 import EPSILON_START, EPSILON_MIN, EPSILON_DECAY, LEARNING_RATE, GAMMA
    from config_v8 import BATCH_SIZE, BUFFER_SIZE, TARGET_UPDATE, Q_VALUE_CLIP, GRAD_CLIP

    agent_config = {
        'epsilon_start': EPSILON_START,
        'epsilon_decay': EPSILON_DECAY,
        'epsilon_min': EPSILON_MIN,
        'lr': LEARNING_RATE,
        'gamma': GAMMA,
        'batch_size': BATCH_SIZE,
        'buffer_size': BUFFER_SIZE,
        'target_update': TARGET_UPDATE,
        'q_value_clip': Q_VALUE_CLIP,
        'grad_clip': GRAD_CLIP,
        'use_scheduler': False,
        'decay_schedule': 'step',
        'total_episodes': episodes,
        'state_channels': 7
    }

    env = SwitchHuntTrainingEnvV8(render=render)
    ghost_agent = ConfigurableDQNAI(agent_config)
    checkpoint_manager = CheckpointManager()

    # 训练统计
    episode_rewards = []
    episode_lengths = []
    episode_catches = []
    episode_correct_dirs = []
    episode_bait_counts = []  # V8.23: 每集引诱成功次数
    episode_phases = []       # V8.23: 各阶段步数占比

    start_time = datetime.now()

    for episode in range(1, episodes + 1):
        ghost_agent.set_episode(episode)

        ghost_state, _ = env.reset()

        # V8.26: 课程学习 — 根据当前阶段配置玩家（两阶段，Phase3已禁用）
        # [V8.26] Phase3已暂时禁用：原Phase3（AI玩家移动）导致鬼学会静止不动
        # 因为玩家AI太傻会主动冲向鬼，鬼无需移动就能抓到玩家，从而学会懈怠。
        if episode <= CURRICULUM['phase1_end']:
            env.game.player.light_charges = 0
            env.game.player.light_charges_max = 0
            env.game.player.ai_enabled = False
            curr_phase_name = "Phase1-Sprint"
        elif episode <= CURRICULUM['phase2_end']:
            # Phase2a: 原Phase2区间（静止玩家+光源引诱学习）
            env.game.player.light_charges = LIGHT_SYSTEM['initial_charges']
            env.game.player.light_charges_max = LIGHT_SYSTEM['max_charges']
            env.game.player.ai_enabled = False
            curr_phase_name = "Phase2a-Bait"
        else:
            # [V8.26] Phase2b: 原Phase3区间现在继续Phase2训练
            # 原计划是Phase3（AI玩家移动），但因玩家AI问题暂时改为静止玩家
            env.game.player.light_charges = LIGHT_SYSTEM['initial_charges']
            env.game.player.light_charges_max = LIGHT_SYSTEM['max_charges']
            env.game.player.ai_enabled = False
            curr_phase_name = "Phase2b-Bait"

        total_reward = 0
        done = False
        step_count = 0
        correct_dir_count = 0
        bait_count = 0
        phase_steps = {'stalk': 0, 'sprint': 0, 'light_active': 0}

        while not done:
            action = ghost_agent.get_action(ghost_state, training=True)

            # V8.23: 使用离散化步骤（1决策=1格间移动）
            next_state, reward, done, info = env.step_discrete(action)

            # 统计正确方向
            if info.get('planned_direction') is not None:
                if action == info['planned_direction']:
                    correct_dir_count += 1

            # 统计引诱成功和阶段分布
            if info.get('bait_triggered'):
                bait_count += 1
            phase_steps[info.get('phase', 'sprint')] = \
                phase_steps.get(info.get('phase', 'sprint'), 0) + 1

            ghost_agent.store_transition(ghost_state, action, reward, next_state, done)
            ghost_agent.learn()

            ghost_state = next_state
            total_reward += reward
            step_count = info['step']

            if render:
                env.render(last_action=action, last_reward=reward)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n[用户] 关闭窗口")
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("\n[用户] 按ESC停止")
                            return

        ghost_agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        episode_catches.append(1 if info['caught'] else 0)
        episode_bait_counts.append(bait_count)

        correct_rate = correct_dir_count / step_count * 100 if step_count > 0 else 0
        episode_correct_dirs.append(correct_rate)
        episode_phases.append(phase_steps)

        # 打印进度
        if episode % print_every == 0:
            # 计算最近print_every轮的统计
            recent_rewards = episode_rewards[-print_every:]
            recent_lengths = episode_lengths[-print_every:]
            recent_catches = episode_catches[-print_every:]
            recent_corrects = episode_correct_dirs[-print_every:]
            recent_baits = episode_bait_counts[-print_every:]

            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            catch_rate = sum(recent_catches) / print_every * 100
            avg_correct = np.mean(recent_corrects)
            avg_bait = np.mean(recent_baits)

            # 额外统计
            max_reward = np.max(recent_rewards)
            min_reward = np.min(recent_rewards)
            std_reward = np.std(recent_rewards)

            stats = ghost_agent.get_stats()

            print("\n" + "=" * 80)
            print(f"[EP {episode:4d}/{episodes}] 训练进度: {episode/episodes*100:.1f}% | 阶段: {curr_phase_name}")
            print("-" * 80)
            print(f"奖励统计 (最近{print_every}轮):")
            print(f"  平均: {avg_reward:8.2f} | 最高: {max_reward:8.2f} | 最低: {min_reward:8.2f} | 标准差: {std_reward:6.2f}")
            print(f"  本回合: {total_reward:8.2f} | 步数: {step_count:4d} | 抓到: {'是' if info['caught'] else '否'}")
            print(f"关键指标:")
            print(f"  抓捕率: {catch_rate:5.1f}% | 方向率: {avg_correct:5.1f}% | 引诱成功: {avg_bait:.1f}/集 | ε: {ghost_agent.epsilon:.3f}")
            print(f"网络状态:")
            print(f"  平均Loss: {stats['avg_loss']:.4f} | Q值: {stats.get('avg_q_value', 0):.2f} | 缓冲区: {len(ghost_agent.memory)}/{ghost_agent.memory.buffer.maxlen}")
            print("=" * 80)
            
            # V8.22: 保存检查点
            if checkpoint_manager.should_save(episode):
                checkpoint_stats = {
                    'episode': episode,
                    'avg_reward': float(avg_reward),
                    'avg_length': float(avg_length),
                    'catch_rate': float(catch_rate),
                    'avg_correct': float(avg_correct),
                    'epsilon': float(ghost_agent.epsilon),
                    'total_episodes': episodes
                }
                checkpoint_manager.save_checkpoint(ghost_agent, episode, checkpoint_stats)
    
    # 保存模型
    print("\n" + "=" * 70)
    print("保存模型...")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    ghost_agent.save('models/ghost_v8.pth')
    np.save('logs/v8_rewards.npy', np.array(episode_rewards))
    np.save('logs/v8_lengths.npy', np.array(episode_lengths))
    np.save('logs/v8_correct_dirs.npy', np.array(episode_correct_dirs))
    np.save('logs/v8_bait_counts.npy', np.array(episode_bait_counts))

    print(f"模型保存到: models/ghost_v8.pth")
    print(f"Final avg reward: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final correct direction rate: {np.mean(episode_correct_dirs[-100:]):.1f}%")
    print(f"Total catch rate: {sum(episode_catches)/len(episode_catches)*100:.1f}%")
    print(f"Final avg bait success: {np.mean(episode_bait_counts[-100:]):.2f}/episode")
    
    # V8.22: 输出检查点信息
    if checkpoint_manager.enabled:
        checkpoints = checkpoint_manager.list_checkpoints()
        if checkpoints:
            print(f"\n[V8.22] 检查点列表（共{len(checkpoints)}个）:")
            for cp in checkpoints:
                print(f"  - {os.path.basename(cp['filepath'])} (EP{cp['episode']})")
            print(f"[V8.22] 如需回滚，可加载: {checkpoint_manager.get_latest_checkpoint()}")
    
    # 生成训练可视化图表
    print("\n生成训练可视化图表...")
    plot_training_results(
        episode_rewards, 
        episode_lengths, 
        episode_catches, 
        episode_correct_dirs,
        save_path='logs/v8_training_results.png'
    )
    
    env.close()


def plot_training_results(rewards, lengths, catches, correct_dirs, save_path='logs/training_results.png'):
    """生成训练过程可视化图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('DQN V8 Training Results', fontsize=16, fontweight='bold')
        
        episodes = range(1, len(rewards) + 1)
        
        # 1. 奖励曲线
        ax1 = axes[0, 0]
        ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
        # 添加移动平均
        window = 10
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(rewards)+1), moving_avg, color='red', linewidth=2, label=f'MA({window})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 步数曲线
        ax2 = axes[0, 1]
        ax2.plot(episodes, lengths, alpha=0.3, color='green')
        if len(lengths) >= window:
            moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window, len(lengths)+1), moving_avg, color='red', linewidth=2, label=f'MA({window})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Lengths')
        ax2.grid(True, alpha=0.3)
        
        # 3. 抓捕率
        ax3 = axes[1, 0]
        # 计算移动抓捕率
        catch_window = 10
        catch_rates = []
        for i in range(len(catches)):
            start = max(0, i - catch_window + 1)
            rate = sum(catches[start:i+1]) / (i - start + 1) * 100
            catch_rates.append(rate)
        ax3.plot(episodes, catch_rates, color='purple', linewidth=2)
        ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Target')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Catch Rate (%)')
        ax3.set_title(f'Catch Rate (MA{catch_window})')
        ax3.set_ylim([0, 105])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 正确方向率
        ax4 = axes[1, 1]
        ax4.plot(episodes, correct_dirs, alpha=0.3, color='orange')
        if len(correct_dirs) >= window:
            moving_avg = np.convolve(correct_dirs, np.ones(window)/window, mode='valid')
            ax4.plot(range(window, len(correct_dirs)+1), moving_avg, color='red', linewidth=2, label=f'MA({window})')
        ax4.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='70% Target')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Correct Direction Rate (%)')
        ax4.set_title('Correct Direction Rate')
        ax4.set_ylim([0, 105])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练图表已保存到: {save_path}")
        plt.close()
        
    except ImportError:
        print("[警告] matplotlib未安装，跳过可视化。安装: pip install matplotlib")
    except Exception as e:
        print(f"[错误] 生成图表失败: {e}")


# 命令行入口
# 用法:
#   python dqn_training_system_v8.py                   # 500轮，无渲染
#   python dqn_training_system_v8.py --episodes 1000   # 1000轮
#   python dqn_training_system_v8.py --render 1        # 带实时渲染
# 也可使用 train.py 获得更友好的提示：python train.py -h
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='DQN Training v8.26 — Two-Phase Ghost Behavior (Phase3 Disabled)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
课程学习阶段（默认500轮）:
  Phase 1 (EP  1-100): 纯冲刺 — 静止玩家，0次光源
  Phase 2a (EP101-250): 引诱学习 — 静止玩家，3次光源，300ms延迟开灯
  Phase 2b (EP251-500): 引诱学习+ — 静止玩家，3次光源，300ms延迟开灯
  [V8.26] 注意: Phase3（AI玩家移动）已暂时禁用
        """
    )
    parser.add_argument('--episodes', '-e', type=int, default=500, help='训练回合数 (默认: 500)')
    parser.add_argument('--print-every', '-p', type=int, default=10, help='打印频率 (默认: 10)')
    parser.add_argument('--render', '-r', type=int, default=1, choices=[0, 1],
                        help='实时渲染 (1=开, 0=关, 默认: 0)')
    args = parser.parse_args()

    train_ghost_v8(episodes=args.episodes, print_every=args.print_every,
                   render=args.render == 1)
