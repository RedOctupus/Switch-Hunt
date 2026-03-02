#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《开关猎杀》游戏 - 迷宫生成与碰撞系统
包含：迷宫生成、墙壁碰撞检测、滑移响应
"""

import pygame
import random
import math
from typing import Tuple, List, Optional

# ==================== 常量定义 ====================
# 迷宫网格大小（必须是奇数，确保有墙壁包围）
GRID_WIDTH = 11      # 网格宽度（格子数）
GRID_HEIGHT = 11     # 网格高度（格子数）
CELL_SIZE = 32       # 每个格子像素大小

# 计算实际屏幕尺寸
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE   # 屏幕宽度（像素）
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE # 屏幕高度（像素）

# 地图元素类型
EMPTY = 0    # 空地
WALL = 1     # 墙壁

# 玩家碰撞箱参数
PLAYER_RADIUS = 12   # 玩家圆形碰撞箱半径（像素）
PLAYER_SPEED = 3     # 玩家移动速度（像素/帧）

# 颜色定义
COLOR_BG = (20, 20, 30)        # 背景颜色（深蓝黑色）
COLOR_WALL = (60, 60, 80)      # 墙壁颜色（深灰色）
COLOR_WALL_BORDER = (100, 100, 120)  # 墙壁边框颜色
COLOR_PLAYER = (100, 200, 100) # 玩家颜色（绿色）
COLOR_PLAYER_BORDER = (150, 255, 150) # 玩家边框颜色


# ==================== 辅助函数 ====================
def grid_to_pixel(grid_x: int, grid_y: int) -> Tuple[int, int]:
    """
    将网格坐标转换为像素坐标（格子中心点）
    
    参数:
        grid_x: 网格X坐标（0到GRID_WIDTH-1）
        grid_y: 网格Y坐标（0到GRID_HEIGHT-1）
    
    返回:
        (pixel_x, pixel_y): 像素坐标（格子中心）
    """
    pixel_x = grid_x * CELL_SIZE + CELL_SIZE // 2  # 计算X方向中心像素
    pixel_y = grid_y * CELL_SIZE + CELL_SIZE // 2  # 计算Y方向中心像素
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
    grid_x = int(pixel_x // CELL_SIZE)  # 整数除法得到网格X
    grid_y = int(pixel_y // CELL_SIZE)  # 整数除法得到网格Y
    # 限制在有效范围内
    grid_x = max(0, min(grid_x, GRID_WIDTH - 1))
    grid_y = max(0, min(grid_y, GRID_HEIGHT - 1))
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
    dx = x2 - x1  # X方向差值
    dy = y2 - y1  # Y方向差值
    return math.sqrt(dx * dx + dy * dy)  # 返回欧几里得距离


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
    return max(min_val, min(max_val, value))


# ==================== 迷宫生成类 ====================
class Map:
    """
    迷宫地图类
    负责迷宫生成、碰撞检测和渲染
    """
    
    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT):
        """
        初始化地图
        
        参数:
            width: 地图宽度（格子数）
            height: 地图高度（格子数）
        """
        self.width = width      # 保存地图宽度
        self.height = height    # 保存地图高度
        # 创建二维数组，初始全部为墙壁
        self.grid = [[WALL for _ in range(width)] for _ in range(height)]
        # 生成迷宫
        self.generate_maze()
    
    def generate_maze(self) -> None:
        """
        使用深度优先回溯算法生成迷宫
        确保所有通道连通，单格宽度
        """
        # 从起点开始（必须是奇数坐标，确保在通道上）
        start_x, start_y = 1, 1
        self.grid[start_y][start_x] = EMPTY  # 标记起点为空地
        
        # 使用栈来记录路径
        stack = [(start_x, start_y)]
        
        # 定义四个方向的移动（上、下、左、右）
        # 每次移动2格，确保挖通墙壁
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        
        # 深度优先搜索生成迷宫
        while stack:
            # 获取当前位置
            current_x, current_y = stack[-1]
            
            # 获取所有未访问的邻居
            neighbors = []
            for dx, dy in directions:
                next_x = current_x + dx
                next_y = current_y + dy
                
                # 检查是否在边界内且是墙壁（未访问）
                if (0 < next_x < self.width - 1 and 
                    0 < next_y < self.height - 1 and 
                    self.grid[next_y][next_x] == WALL):
                    neighbors.append((next_x, next_y, dx, dy))
            
            if neighbors:
                # 随机选择一个邻居
                next_x, next_y, dx, dy = random.choice(neighbors)
                
                # 挖通当前位置到邻居之间的墙壁
                wall_x = current_x + dx // 2
                wall_y = current_y + dy // 2
                self.grid[wall_y][wall_x] = EMPTY
                
                # 标记邻居为空地
                self.grid[next_y][next_x] = EMPTY
                
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
        return self.grid[grid_y][grid_x] == WALL
    
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
        # 收集所有空地位置
        empty_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == EMPTY:
                    empty_positions.append((x, y))
        
        # 随机返回一个空地位置
        return random.choice(empty_positions)
    
    def get_wall_rects_around(self, pixel_x: float, pixel_y: float, radius: float) -> List[pygame.Rect]:
        """
        获取指定位置周围的所有墙壁矩形
        用于碰撞检测优化（只检测附近的墙壁）
        
        参数:
            pixel_x: 像素X坐标
            pixel_y: 像素Y坐标
            radius: 检测半径
        
        返回:
            墙壁矩形列表
        """
        wall_rects = []
        # 计算需要检测的网格范围（扩大一点确保不遗漏）
        min_grid_x = int((pixel_x - radius) // CELL_SIZE) - 1
        max_grid_x = int((pixel_x + radius) // CELL_SIZE) + 1
        min_grid_y = int((pixel_y - radius) // CELL_SIZE) - 1
        max_grid_y = int((pixel_y + radius) // CELL_SIZE) + 1
        
        # 限制在有效范围内
        min_grid_x = max(0, min_grid_x)
        max_grid_x = min(self.width - 1, max_grid_x)
        min_grid_y = max(0, min_grid_y)
        max_grid_y = min(self.height - 1, max_grid_y)
        
        # 收集范围内所有墙壁的矩形
        for gy in range(min_grid_y, max_grid_y + 1):
            for gx in range(min_grid_x, max_grid_x + 1):
                if self.grid[gy][gx] == WALL:
                    # 创建墙壁矩形（像素坐标）
                    wall_rect = pygame.Rect(
                        gx * CELL_SIZE,      # 左上角X
                        gy * CELL_SIZE,      # 左上角Y
                        CELL_SIZE,           # 宽度
                        CELL_SIZE            # 高度
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
    
    def check_collision(self, pixel_x: float, pixel_y: float, radius: float) -> Tuple[bool, List[Tuple[pygame.Rect, float, float]]]:
        """
        检查圆形碰撞箱与所有墙壁的碰撞
        
        参数:
            pixel_x: 圆心X坐标
            pixel_y: 圆心Y坐标
            radius: 圆半径
        
        返回:
            (是否碰撞, [(墙壁矩形, 最近点X, 最近点Y), ...])
        """
        collisions = []
        
        # 获取周围的所有墙壁
        wall_rects = self.get_wall_rects_around(pixel_x, pixel_y, radius)
        
        # 检测与每个墙壁的碰撞
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
        is_colliding, collisions = self.check_collision(target_x, target_y, radius)
        
        if not is_colliding:
            # 没有碰撞，直接返回目标位置
            return (target_x, target_y)
        
        # 有碰撞，尝试分别移动X和Y方向
        # 先尝试X方向
        test_x = pixel_x + dx
        test_y = pixel_y
        collide_x, _ = self.check_collision(test_x, test_y, radius)
        
        # 再尝试Y方向
        test_x = pixel_x
        test_y = pixel_y + dy
        collide_y, _ = self.check_collision(test_x, test_y, radius)
        
        # 根据碰撞情况决定滑动方向
        new_x, new_y = pixel_x, pixel_y
        
        if not collide_x:
            # X方向可以移动
            new_x = pixel_x + dx
        else:
            # X方向有碰撞，处理X方向的滑移
            for wall_rect, closest_x, closest_y in collisions:
                if abs(dy) < 0.01:  # 主要是水平移动
                    if pixel_x < wall_rect.centerx:
                        # 在墙壁左侧，向左推
                        new_x = wall_rect.left - radius - 0.1
                    else:
                        # 在墙壁右侧，向右推
                        new_x = wall_rect.right + radius + 0.1
        
        if not collide_y:
            # Y方向可以移动
            new_y = pixel_y + dy
        else:
            # Y方向有碰撞，处理Y方向的滑移
            for wall_rect, closest_x, closest_y in collisions:
                if abs(dx) < 0.01:  # 主要是垂直移动
                    if pixel_y < wall_rect.centery:
                        # 在墙壁上方，向上推
                        new_y = wall_rect.top - radius - 0.1
                    else:
                        # 在墙壁下方，向下推
                        new_y = wall_rect.bottom + radius + 0.1
        
        # 如果两个方向都有碰撞，尝试找到不碰撞的位置
        if collide_x and collide_y:
            # 尝试沿着墙壁滑动（优先保持移动方向）
            if abs(dx) > abs(dy):
                # 主要是水平移动，尝试只移动Y
                if not collide_y:
                    new_x = pixel_x
                    new_y = pixel_y + dy
                else:
                    # 两个方向都不行，保持原位
                    new_x, new_y = pixel_x, pixel_y
            else:
                # 主要是垂直移动，尝试只移动X
                if not collide_x:
                    new_x = pixel_x + dx
                    new_y = pixel_y
                else:
                    # 两个方向都不行，保持原位
                    new_x, new_y = pixel_x, pixel_y
        
        return (new_x, new_y)
    
    def render(self, screen: pygame.Surface) -> None:
        """
        渲染迷宫到屏幕
        
        参数:
            screen: Pygame屏幕对象
        """
        # 遍历所有格子
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == WALL:
                    # 计算墙壁矩形位置
                    wall_rect = pygame.Rect(
                        x * CELL_SIZE,
                        y * CELL_SIZE,
                        CELL_SIZE,
                        CELL_SIZE
                    )
                    # 绘制墙壁填充
                    pygame.draw.rect(screen, COLOR_WALL, wall_rect)
                    # 绘制墙壁边框
                    pygame.draw.rect(screen, COLOR_WALL_BORDER, wall_rect, 2)


# ==================== 玩家类 ====================
class Player:
    """
    玩家类
    处理玩家移动和碰撞
    """
    
    def __init__(self, x: float, y: float, radius: float = PLAYER_RADIUS):
        """
        初始化玩家
        
        参数:
            x: 初始X坐标（像素）
            y: 初始Y坐标（像素）
            radius: 碰撞箱半径
        """
        self.x = x              # 玩家X坐标
        self.y = y              # 玩家Y坐标
        self.radius = radius    # 碰撞箱半径
        self.speed = PLAYER_SPEED  # 移动速度
    
    def move(self, dx: float, dy: float, game_map: Map) -> None:
        """
        移动玩家，处理碰撞
        
        参数:
            dx: X方向移动距离
            dy: Y方向移动距离
            game_map: 地图对象
        """
        # 使用滑移碰撞响应
        new_x, new_y = game_map.resolve_collision_slide(
            self.x, self.y, self.radius, dx, dy
        )
        
        # 限制在屏幕范围内
        self.x = clamp(new_x, self.radius, SCREEN_WIDTH - self.radius)
        self.y = clamp(new_y, self.radius, SCREEN_HEIGHT - self.radius)
    
    def render(self, screen: pygame.Surface) -> None:
        """
        渲染玩家
        
        参数:
            screen: Pygame屏幕对象
        """
        # 绘制玩家圆形
        pygame.draw.circle(screen, COLOR_PLAYER, (int(self.x), int(self.y)), self.radius)
        # 绘制边框
        pygame.draw.circle(screen, COLOR_PLAYER_BORDER, (int(self.x), int(self.y)), self.radius, 2)
        # 绘制方向指示（小圆点）
        pygame.draw.circle(screen, COLOR_PLAYER_BORDER, 
                          (int(self.x + self.radius * 0.5), int(self.y)), 3)


# ==================== 游戏主类 ====================
class Game:
    """
    游戏主类
    管理游戏循环和状态
    """
    
    def __init__(self):
        """初始化游戏"""
        # 初始化Pygame
        pygame.init()
        
        # 创建窗口
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("《开关猎杀》- 迷宫生成与碰撞系统测试")
        
        # 创建时钟控制帧率
        self.clock = pygame.time.Clock()
        
        # 创建字体用于显示信息
        self.font = pygame.font.SysFont("simhei", 16)
        
        # 生成迷宫
        self.game_map = Map(GRID_WIDTH, GRID_HEIGHT)
        
        # 在随机空地位置创建玩家
        start_grid = self.game_map.get_random_empty_position()
        start_pixel = grid_to_pixel(start_grid[0], start_grid[1])
        self.player = Player(start_pixel[0], start_pixel[1])
        
        # 按键状态
        self.keys_pressed = {
            pygame.K_UP: False,
            pygame.K_DOWN: False,
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False,
            pygame.K_w: False,
            pygame.K_s: False,
            pygame.K_a: False,
            pygame.K_d: False,
        }
        
        # 运行状态
        self.running = True
    
    def handle_events(self) -> None:
        """处理输入事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # 关闭窗口
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                # 按键按下
                if event.key in self.keys_pressed:
                    self.keys_pressed[event.key] = True
                
                # R键重新生成迷宫
                if event.key == pygame.K_r:
                    self.game_map = Map(GRID_WIDTH, GRID_HEIGHT)
                    start_grid = self.game_map.get_random_empty_position()
                    start_pixel = grid_to_pixel(start_grid[0], start_grid[1])
                    self.player.x = start_pixel[0]
                    self.player.y = start_pixel[1]
                
                # ESC键退出
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            
            elif event.type == pygame.KEYUP:
                # 按键释放
                if event.key in self.keys_pressed:
                    self.keys_pressed[event.key] = False
    
    def update(self) -> None:
        """更新游戏状态"""
        # 计算移动方向
        dx, dy = 0, 0
        
        # 上下移动（W/S或方向键）
        if self.keys_pressed[pygame.K_UP] or self.keys_pressed[pygame.K_w]:
            dy -= self.player.speed
        if self.keys_pressed[pygame.K_DOWN] or self.keys_pressed[pygame.K_s]:
            dy += self.player.speed
        
        # 左右移动（A/D或方向键）
        if self.keys_pressed[pygame.K_LEFT] or self.keys_pressed[pygame.K_a]:
            dx -= self.player.speed
        if self.keys_pressed[pygame.K_RIGHT] or self.keys_pressed[pygame.K_d]:
            dx += self.player.speed
        
        # 如果有移动输入，移动玩家
        if dx != 0 or dy != 0:
            self.player.move(dx, dy, self.game_map)
    
    def render(self) -> None:
        """渲染游戏画面"""
        # 填充背景
        self.screen.fill(COLOR_BG)
        
        # 渲染迷宫
        self.game_map.render(self.screen)
        
        # 渲染玩家
        self.player.render(self.screen)
        
        # 渲染调试信息
        self.render_debug_info()
        
        # 更新显示
        pygame.display.flip()
    
    def render_debug_info(self) -> None:
        """渲染调试信息"""
        # 玩家位置信息
        grid_pos = pixel_to_grid(self.player.x, self.player.y)
        info_texts = [
            f"玩家像素: ({int(self.player.x)}, {int(self.player.y)})",
            f"玩家网格: ({grid_pos[0]}, {grid_pos[1]})",
            f"碰撞半径: {self.player.radius}px",
            f"按R重新生成迷宫",
            f"按ESC退出",
        ]
        
        # 绘制信息背景
        info_bg = pygame.Rect(5, 5, 200, len(info_texts) * 20 + 10)
        pygame.draw.rect(self.screen, (0, 0, 0, 128), info_bg)
        
        # 绘制文字
        for i, text in enumerate(info_texts):
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, 10 + i * 20))
    
    def run(self) -> None:
        """运行游戏主循环"""
        while self.running:
            # 处理事件
            self.handle_events()
            
            # 更新状态
            self.update()
            
            # 渲染画面
            self.render()
            
            # 控制帧率（60FPS）
            self.clock.tick(60)
        
        # 退出Pygame
        pygame.quit()


# ==================== 主程序入口 ====================
def main():
    """主函数"""
    print("=" * 50)
    print("《开关猎杀》- 迷宫生成与碰撞系统测试")
    print("=" * 50)
    print("操作说明:")
    print("  - 方向键或WASD: 移动玩家")
    print("  - R键: 重新生成迷宫")
    print("  - ESC键: 退出游戏")
    print("=" * 50)
    
    # 创建并运行游戏
    game = Game()
    game.run()
    
    print("游戏已退出")


if __name__ == "__main__":
    main()
