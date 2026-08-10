"""V7 UISystem：菜单 / HUD / 胜负界面。"""
from __future__ import annotations

from typing import Optional

import pygame

from switch_hunt.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    COLOR_BLACK, COLOR_WHITE, COLOR_GRAY, COLOR_DARK_GRAY,
    COLOR_YELLOW, COLOR_BLUE, COLOR_GREEN, COLOR_RED, COLOR_ORANGE, COLOR_GOLD, COLOR_CYAN,
)
from switch_hunt.enums import GameState

class UISystem:
    """
    UI系统
    负责HUD、菜单和作弊模式界面
    """

    def __init__(self, screen_width: int, screen_height: int):
        """
        初始化UI系统

        参数:
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

        # 字体 - 尝试加载系统中文字体，失败则使用默认字体
        self.font_large = self._load_font(72)
        self.font_medium = self._load_font(48)
        self.font_small = self._load_font(36)
        
        # 如果系统没有中文字体，使用英文文本作为备选
        self.use_english = self.font_large is None or not self._check_font_support()
        if self.use_english:
            self.font_large = pygame.font.Font(None, 72)
            self.font_medium = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 36)
        
        # 菜单选项（根据字体支持决定语言）
        self.ai_mode = False  # AI演示模式
        self.difficulty = 'normal'  # 难度: easy, normal, hard
        
        if self.use_english:
            self.menu_options = [
                "Start Game", 
                "Difficulty: Normal",
                "AI Demo: OFF",
                "Cheat Mode: OFF", 
                "Quit"
            ]
        else:
            self.menu_options = [
                "开始游戏", 
                "难度: 普通",
                "AI演示: 关",
                "作弊模式: 关", 
                "退出"
            ]
        self.selected_option = 0

        # 作弊模式
        self.cheat_mode = False

        # HUD位置
        self.hud_margin = 20
        self.energy_bar_width = 200
        self.energy_bar_height = 20
    
    def _load_font(self, size: int) -> Optional[pygame.font.Font]:
        """尝试加载支持中文的字体"""
        # 常见的Windows中文字体路径
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑粗体
        ]
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    return pygame.font.Font(font_path, size)
            except:
                continue
        return None
    
    def _check_font_support(self) -> bool:
        """检查字体是否支持中文字符"""
        try:
            test_surface = self.font_large.render("测试", True, COLOR_WHITE)
            return test_surface.get_width() > 20
        except:
            return False

    def handle_menu_input(self, event: pygame.event.Event) -> Optional[str]:
        """
        处理菜单输入

        参数:
            event: Pygame事件对象

        返回:
            选择的操作，或None
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_option = (self.selected_option - 1) % len(self.menu_options)
            elif event.key == pygame.K_DOWN:
                self.selected_option = (self.selected_option + 1) % len(self.menu_options)
            elif event.key == pygame.K_RETURN:
                if self.selected_option == 0:
                    return "start"
                elif self.selected_option == 1:
                    # 切换难度: normal -> easy -> hard -> normal
                    if self.difficulty == 'normal':
                        self.difficulty = 'easy'
                    elif self.difficulty == 'easy':
                        self.difficulty = 'hard'
                    else:
                        self.difficulty = 'normal'
                    
                    if self.use_english:
                        diff_text = self.difficulty.capitalize()
                        self.menu_options[1] = f"Difficulty: {diff_text}"
                    else:
                        diff_map = {'easy': '简单', 'normal': '普通', 'hard': '困难'}
                        self.menu_options[1] = f"难度: {diff_map[self.difficulty]}"
                        
                elif self.selected_option == 2:
                    # 切换AI演示模式
                    self.ai_mode = not self.ai_mode
                    if self.use_english:
                        self.menu_options[2] = f"AI Demo: {'ON' if self.ai_mode else 'OFF'}"
                    else:
                        self.menu_options[2] = f"AI演示: {'开' if self.ai_mode else '关'}"
                        
                elif self.selected_option == 3:
                    self.cheat_mode = not self.cheat_mode
                    if self.use_english:
                        self.menu_options[3] = f"Cheat Mode: {'ON' if self.cheat_mode else 'OFF'}"
                    else:
                        self.menu_options[3] = f"作弊模式: {'开' if self.cheat_mode else '关'}"
                        
                elif self.selected_option == 4:
                    return "quit"
        return None

    def render_menu(self, screen: pygame.Surface):
        """
        渲染主菜单

        参数:
            screen: Pygame屏幕表面
        """
        screen.fill(COLOR_BLACK)

        # 标题
        if self.use_english:
            title = self.font_large.render("Switch Hunt", True, COLOR_GOLD)
            subtitle_text = ""
        else:
            title = self.font_large.render("开关猎杀", True, COLOR_GOLD)
            subtitle_text = "Switch Hunt"
        title_rect = title.get_rect(center=(self.screen_width // 2, 150))
        screen.blit(title, title_rect)

        # 副标题
        if subtitle_text:
            subtitle = self.font_small.render(subtitle_text, True, COLOR_GRAY)
            subtitle_rect = subtitle.get_rect(center=(self.screen_width // 2, 210))
            screen.blit(subtitle, subtitle_rect)

        # 菜单选项
        for i, option in enumerate(self.menu_options):
            color = COLOR_YELLOW if i == self.selected_option else COLOR_WHITE
            text = self.font_medium.render(option, True, color)
            rect = text.get_rect(center=(self.screen_width // 2, 350 + i * 60))
            screen.blit(text, rect)

        # 操作提示
        if self.use_english:
            hint_text = "Use Arrow Keys to select, Enter to confirm"
        else:
            hint_text = "使用方向键选择，回车确认"
        hint = self.font_small.render(hint_text, True, COLOR_GRAY)
        hint_rect = hint.get_rect(center=(self.screen_width // 2, self.screen_height - 100))
        screen.blit(hint, hint_rect)

    def render_hud(self, screen: pygame.Surface, player: Player,
                   treasures_collected: int, total_treasures: int):
        """
        渲染HUD

        参数:
            screen: Pygame屏幕表面
            player: 玩家对象
            treasures_collected: 已收集宝藏数
            total_treasures: 总宝藏数
        """
        # 宝藏计数
        if self.use_english:
            treasure_label = "Treasures"
        else:
            treasure_label = "宝藏"
        treasure_text = self.font_small.render(
            f"{treasure_label}: {treasures_collected}/{total_treasures}", True, COLOR_GOLD)
        screen.blit(treasure_text, (self.hud_margin, self.hud_margin))

        # 能量条背景
        bar_x = self.hud_margin
        bar_y = self.hud_margin + 40
        pygame.draw.rect(screen, COLOR_DARK_GRAY,
                        (bar_x, bar_y, self.energy_bar_width, self.energy_bar_height))

        # 能量条填充
        energy_ratio = player.energy / player.max_energy
        energy_width = int(self.energy_bar_width * energy_ratio)

        if energy_ratio > 0.5:
            energy_color = COLOR_GREEN
        elif energy_ratio > 0.25:
            energy_color = COLOR_YELLOW
        else:
            energy_color = COLOR_RED

        pygame.draw.rect(screen, energy_color,
                        (bar_x, bar_y, energy_width, self.energy_bar_height))

        # 能量条边框
        pygame.draw.rect(screen, COLOR_WHITE,
                        (bar_x, bar_y, self.energy_bar_width, self.energy_bar_height), 2)

        # 能量数值
        if self.use_english:
            energy_label = "Energy"
        else:
            energy_label = "能量"
        energy_text = self.font_small.render(
            f"{energy_label}: {int(player.energy)}/{player.max_energy}", True, COLOR_WHITE)
        screen.blit(energy_text, (bar_x + self.energy_bar_width + 10, bar_y - 2))

        # 光源模式指示
        if self.use_english:
            light_text = "Enhanced Light" if player.is_enhanced_light() else "Normal Light"
        else:
            light_text = "强化光源" if player.is_enhanced_light() else "普通光源"
        light_color = COLOR_ORANGE if player.is_enhanced_light() else COLOR_YELLOW
        light_surface = self.font_small.render(light_text, True, light_color)
        screen.blit(light_surface, (bar_x, bar_y + 30))

        # 作弊模式指示
        if self.cheat_mode:
            if self.use_english:
                cheat_label = "[Cheat Mode ON]"
            else:
                cheat_label = "[作弊模式开启]"
            cheat_text = self.font_small.render(cheat_label, True, COLOR_RED)
            screen.blit(cheat_text, (self.screen_width - 200, self.hud_margin))

    def render_pause(self, screen: pygame.Surface):
        """
        渲染暂停界面

        参数:
            screen: Pygame屏幕表面
        """
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.fill(COLOR_BLACK)
        overlay.set_alpha(180)
        screen.blit(overlay, (0, 0))

        if self.use_english:
            pause_text = "Game Paused"
        else:
            pause_text = "游戏暂停"
        text = self.font_large.render(pause_text, True, COLOR_WHITE)
        rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        screen.blit(text, rect)

        if self.use_english:
            hint_text = "Press P to continue"
        else:
            hint_text = "按 P 继续游戏"
        hint = self.font_small.render(hint_text, True, COLOR_GRAY)
        hint_rect = hint.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 60))
        screen.blit(hint, hint_rect)

    def render_victory(self, screen: pygame.Surface):
        """
        渲染胜利界面

        参数:
            screen: Pygame屏幕表面
        """
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.fill(COLOR_BLACK)
        overlay.set_alpha(200)
        screen.blit(overlay, (0, 0))

        if self.use_english:
            victory_text = "Victory!"
        else:
            victory_text = "恭喜胜利！"
        text = self.font_large.render(victory_text, True, COLOR_GOLD)
        rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        screen.blit(text, rect)

        if self.use_english:
            hint_text = "Press R to restart, ESC for menu"
        else:
            hint_text = "按 R 重新开始，按 ESC 返回菜单"
        hint = self.font_small.render(hint_text, True, COLOR_WHITE)
        hint_rect = hint.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 30))
        screen.blit(hint, hint_rect)

    def render_game_over(self, screen: pygame.Surface):
        """
        渲染失败界面

        参数:
            screen: Pygame屏幕表面
        """
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.fill(COLOR_BLACK)
        overlay.set_alpha(200)
        screen.blit(overlay, (0, 0))

        if self.use_english:
            game_over_text = "Game Over"
        else:
            game_over_text = "游戏结束"
        text = self.font_large.render(game_over_text, True, COLOR_RED)
        rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        screen.blit(text, rect)

        if self.use_english:
            hint_text = "Press R to restart, ESC for menu"
        else:
            hint_text = "按 R 重新开始，按 ESC 返回菜单"
        hint = self.font_small.render(hint_text, True, COLOR_WHITE)
        hint_rect = hint.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 30))
        screen.blit(hint, hint_rect)

    def render_help(self, screen: pygame.Surface):
        """
        渲染帮助信息

        参数:
            screen: Pygame屏幕表面
        """
        # 获取当前设置
        difficulty = self.difficulty.capitalize() if self.use_english else \
                    {'easy': '简单', 'normal': '普通', 'hard': '困难'}.get(self.difficulty, '普通')
        ai_mode = "ON" if self.ai_mode else "OFF"
        
        if self.use_english:
            help_lines = [
                f"Controls: | Difficulty: {difficulty} | AI Demo: {ai_mode}",
                "WASD/Arrow - Move | Space - Light | P - Pause | ESC - Menu",
            ]
        else:
            help_lines = [
                f"操作: WASD移动 空格光源 | 难度: {difficulty} | AI演示: {'开' if self.ai_mode else '关'}",
                "P暂停 ESC返回菜单",
            ]

        y_offset = self.screen_height - 100
        for line in help_lines:
            text = self.font_small.render(line, True, COLOR_GRAY)
            screen.blit(text, (self.hud_margin, y_offset))
            y_offset += 25


# =============================================================================
# 第十二部分：游戏管理器
# =============================================================================
