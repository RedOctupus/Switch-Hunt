"""统一视觉主题：配色、字体缓存、通用绘制工具。"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple, Optional, List

import math
import pygame

# ---------------------------------------------------------------------------
# 调色板 — 深夜迷宫 + 暖琥珀光源
# ---------------------------------------------------------------------------
BG_DEEP = (8, 10, 16)
BG_MID = (14, 18, 28)
BG_PANEL = (18, 24, 36)
BG_PANEL_EDGE = (48, 62, 82)

WALL_FILL = (58, 72, 92)
WALL_TOP = (78, 96, 118)
WALL_EDGE = (36, 44, 58)
FLOOR_A = (22, 26, 36)
FLOOR_B = (26, 30, 42)
FLOOR_LINE = (32, 38, 52)

ACCENT_WARM = (255, 176, 64)       # 琥珀主强调
ACCENT_WARM_DIM = (200, 130, 40)
ACCENT_GOLD = (242, 201, 76)
ACCENT_DANGER = (220, 64, 72)
ACCENT_SAFE = (72, 196, 140)
ACCENT_INFO = (88, 196, 210)
ACCENT_STUN = (110, 168, 255)

TEXT_PRIMARY = (236, 240, 248)
TEXT_SECONDARY = (148, 160, 180)
TEXT_MUTED = (96, 108, 128)
TEXT_TITLE = (255, 210, 120)

PLAYER_BODY = (70, 150, 255)
PLAYER_CORE = (180, 220, 255)
PLAYER_OUTLINE = (40, 90, 180)
GHOST_BODY = (210, 48, 58)
GHOST_CORE = (255, 120, 110)
GHOST_OUTLINE = (255, 200, 200)
GHOST_STUN_BODY = (90, 140, 255)
GHOST_STUN_CORE = (180, 210, 255)
TREASURE_CORE = (255, 220, 90)
TREASURE_EDGE = (255, 170, 40)
TREASURE_GLOW = (255, 200, 80)

LIGHT_WARM = (255, 200, 120)
LIGHT_ACTIVE = (255, 140, 50)

Color = Tuple[int, int, int]
ColorA = Tuple[int, int, int, int]


def lerp_color(a: Color, b: Color, t: float) -> Color:
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def with_alpha(color: Color, alpha: int) -> ColorA:
    return (color[0], color[1], color[2], max(0, min(255, alpha)))


@lru_cache(maxsize=32)
def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    """加载可显示中文的字体。

    优先按文件路径加载（绕过部分 Windows 上 pygame.SysFont 扫描失败导致乱码），
    再回退 SysFont，最后才用默认字体。
    """
    windir = os.environ.get("WINDIR", r"C:\Windows")
    fonts_dir = os.path.join(windir, "Fonts")
    file_candidates = (
        ("msyhbd.ttc", True),   # 微软雅黑 Bold
        ("msyh.ttc", False),    # 微软雅黑
        ("simhei.ttf", False),  # 黑体
        ("msjh.ttc", False),    # 微软正黑体
        ("Deng.ttf", False),    # 等线
        ("simsun.ttc", False),  # 宋体
        ("simkai.ttf", False),
        ("simfang.ttf", False),
    )

    # bold 时优先粗体文件，否则普通文件在前
    ordered = sorted(file_candidates, key=lambda item: 0 if item[1] == bold else 1)
    for filename, _is_bold in ordered:
        path = os.path.join(fonts_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            return pygame.font.Font(path, size)
        except Exception:
            continue

    # SysFont 在部分环境会因字体枚举异常而失败/回退到无中文默认字体
    for name in ("microsoftyahei", "msyh", "simhei", "simsun", "dengxian"):
        try:
            font = pygame.font.SysFont(name, size, bold=bold)
            # 探测是否真能渲染中文（默认字体常画成方框/空白）
            probe = font.render("测", True, (255, 255, 255))
            if probe.get_width() > 2:
                return font
        except Exception:
            continue

    return pygame.font.Font(None, size)


def draw_vertical_gradient(
    surface: pygame.Surface,
    top: Color,
    bottom: Color,
    rect: Optional[pygame.Rect] = None,
) -> None:
    """绘制纵向渐变。"""
    target = rect or surface.get_rect()
    h = max(target.height, 1)
    for y in range(target.height):
        c = lerp_color(top, bottom, y / h)
        pygame.draw.line(
            surface, c,
            (target.left, target.top + y),
            (target.right - 1, target.top + y),
        )


def draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    fill: ColorA = (18, 24, 36, 210),
    border: Color = BG_PANEL_EDGE,
    radius: int = 12,
    border_width: int = 1,
) -> None:
    """半透明圆角面板。"""
    panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(panel, fill, panel.get_rect(), border_radius=radius)
    if border_width > 0:
        pygame.draw.rect(panel, border, panel.get_rect(), border_width, border_radius=radius)
    surface.blit(panel, rect.topleft)


def draw_text_centered(
    surface: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    color: Color,
    center: Tuple[int, int],
    shadow: bool = True,
    shadow_color: Color = (0, 0, 0),
) -> pygame.Rect:
    """居中绘制文字，可选阴影。"""
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(center=center)
    if shadow:
        sh = font.render(text, True, shadow_color)
        surface.blit(sh, (rect.x + 2, rect.y + 2))
    surface.blit(rendered, rect)
    return rect


def draw_soft_glow(
    surface: pygame.Surface,
    center: Tuple[int, int],
    radius: int,
    color: Color,
    strength: float = 0.55,
    rings: int = 10,
) -> None:
    """多层半透明圆模拟柔光。"""
    if radius <= 0:
        return
    glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    cx = cy = radius
    for i in range(rings, 0, -1):
        t = i / rings
        a = int(255 * strength * (1.0 - t) ** 2)
        r = int(radius * t)
        if r > 0 and a > 0:
            pygame.draw.circle(glow, with_alpha(color, a), (cx, cy), r)
    surface.blit(glow, (center[0] - radius, center[1] - radius))


def make_radial_light_mask(radius: int, soft_edge: float = 0.55) -> pygame.Surface:
    """生成径向光照遮罩（白=清除迷雾）。缓存友好：按半径生成。"""
    size = max(radius * 2, 2)
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = radius
    inner = int(radius * (1.0 - soft_edge))
    for r in range(radius, 0, -1):
        if r <= inner:
            alpha = 255
        else:
            t = (r - inner) / max(radius - inner, 1)
            alpha = int(255 * (1.0 - t) ** 1.6)
        pygame.draw.circle(mask, (255, 255, 255, alpha), (cx, cy), r)
    return mask


@lru_cache(maxsize=16)
def cached_light_mask(radius: int) -> pygame.Surface:
    return make_radial_light_mask(radius)


def draw_menu_ambiance(surface: pygame.Surface, t: float) -> None:
    """主菜单背景氛围：渐变 + 暗迷宫剪影 + 漂浮光点。"""
    w, h = surface.get_size()
    draw_vertical_gradient(surface, BG_DEEP, (16, 22, 38))

    # 远景迷宫网格剪影
    tile = 40
    maze = pygame.Surface((w, h), pygame.SRCALPHA)
    for gy in range(0, h, tile):
        for gx in range(0, w, tile):
            # 伪随机墙块（确定性）
            n = ((gx * 37 + gy * 91) ^ 0xA5) % 7
            if n < 3:
                shade = 18 + (n * 6)
                pygame.draw.rect(
                    maze, (shade, shade + 4, shade + 12, 55),
                    pygame.Rect(gx + 2, gy + 2, tile - 4, tile - 4),
                    border_radius=4,
                )
    surface.blit(maze, (0, 0))

    # 中央暖光脉冲
    pulse = 0.5 + 0.5 * math.sin(t * 1.4)
    glow_r = int(180 + 40 * pulse)
    draw_soft_glow(surface, (w // 2, int(h * 0.28)), glow_r, ACCENT_WARM, strength=0.18 + 0.08 * pulse, rings=14)

    # 漂浮微尘
    for i in range(18):
        seed = i * 97.3
        x = (math.sin(t * 0.35 + seed) * 0.5 + 0.5) * w
        y = (math.cos(t * 0.28 + seed * 1.3) * 0.5 + 0.5) * h
        a = int(40 + 50 * (0.5 + 0.5 * math.sin(t * 2 + seed)))
        pygame.draw.circle(surface, with_alpha(ACCENT_GOLD, a), (int(x), int(y)), 2)


# 操作教程条目：(按键, 说明)
CONTROL_GUIDE: List[Tuple[str, str]] = [
    ("WASD / 方向键", "移动"),
    ("空格", "强化光源(5格)，照射鬼渐变蓝至定身"),
    ("P", "暂停 / 继续"),
    ("ESC", "返回菜单"),
    ("F1", "作弊全图视野"),
    ("F2", "玩家 AI 演示"),
    ("F3", "显示鬼寻路路径"),
    ("F4", "开关音效 / 配乐"),
]

GAMEPLAY_TIPS: List[str] = [
    "收集全部宝藏即可通关；被鬼碰到则失败。",
    "强化光源半径 5 格，照射下的鬼会逐渐变蓝。",
    "变蓝满格后鬼被定身；拾取宝藏可重置开灯次数。",
]


def draw_controls_guide(
    surface: pygame.Surface,
    rect: pygame.Rect,
    *,
    title: str = "操作教程",
    show_tips: bool = True,
) -> None:
    """绘制操作教程面板。"""
    draw_panel(surface, rect, fill=(12, 16, 26, 210), border=BG_PANEL_EDGE, radius=14)

    title_font = get_font(24, bold=True)
    key_font = get_font(18)
    tip_font = get_font(16)

    title_surf = title_font.render(title, True, ACCENT_GOLD)
    surface.blit(title_surf, (rect.left + 18, rect.top + 14))
    pygame.draw.line(
        surface, ACCENT_WARM_DIM,
        (rect.left + 18, rect.top + 44),
        (rect.right - 18, rect.top + 44),
        1,
    )

    y = rect.top + 56
    col_key_x = rect.left + 20
    col_desc_x = rect.left + 160
    for key, desc in CONTROL_GUIDE:
        key_s = key_font.render(key, True, ACCENT_WARM)
        desc_s = key_font.render(desc, True, TEXT_PRIMARY)
        surface.blit(key_s, (col_key_x, y))
        surface.blit(desc_s, (col_desc_x, y))
        y += 26

    if show_tips:
        y += 8
        pygame.draw.line(
            surface, (40, 50, 68),
            (rect.left + 18, y),
            (rect.right - 18, y),
            1,
        )
        y += 12
        tip_title = tip_font.render("玩法提示", True, TEXT_SECONDARY)
        surface.blit(tip_title, (col_key_x, y))
        y += 22
        for tip in GAMEPLAY_TIPS:
            tip_s = tip_font.render(tip, True, TEXT_MUTED)
            surface.blit(tip_s, (col_key_x, y))
            y += 20


def draw_hud_controls_bar(surface: pygame.Surface) -> None:
    """局内底部简要操作条。"""
    w, h = surface.get_size()
    bar = pygame.Rect(14, h - 42, w - 28, 30)
    draw_panel(surface, bar, fill=(10, 14, 22, 180), border=(40, 50, 68), radius=8)
    font = get_font(16)
    text = "WASD 移动   空格 开灯(5格/渐变蓝)   P 暂停   ESC 菜单   F1 全图   F4 音效"
    surf = font.render(text, True, TEXT_SECONDARY)
    surface.blit(surf, surf.get_rect(center=bar.center))


def draw_overlay_card(
    surface: pygame.Surface,
    title: str,
    subtitle: str,
    title_color: Color,
    hints: Optional[List[str]] = None,
) -> None:
    """暂停/胜负等居中卡片。"""
    w, h = surface.get_size()
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((4, 6, 12, 190))
    surface.blit(overlay, (0, 0))

    card_w, card_h = 520, 220 if not hints else 260
    card = pygame.Rect((w - card_w) // 2, (h - card_h) // 2, card_w, card_h)
    draw_panel(surface, card, fill=(16, 22, 34, 230), border=BG_PANEL_EDGE, radius=16)

    # 顶部装饰线
    pygame.draw.line(
        surface, title_color,
        (card.left + 40, card.top + 28),
        (card.right - 40, card.top + 28),
        2,
    )

    title_font = get_font(52, bold=True)
    sub_font = get_font(24)
    draw_text_centered(surface, title, title_font, title_color, (w // 2, card.centery - 20))
    draw_text_centered(surface, subtitle, sub_font, TEXT_SECONDARY, (w // 2, card.centery + 40), shadow=False)

    if hints:
        hint_font = get_font(20)
        for i, line in enumerate(hints):
            draw_text_centered(
                surface, line, hint_font, TEXT_MUTED,
                (w // 2, card.bottom - 36 + i * 22), shadow=False,
            )


def draw_treasure_icon(surface: pygame.Surface, x: int, y: int, size: int = 8, lit: bool = True) -> None:
    """HUD 用小菱形宝藏图标。"""
    color = TREASURE_CORE if lit else TEXT_MUTED
    edge = TREASURE_EDGE if lit else TEXT_MUTED
    points = [
        (x, y - size),
        (x + size, y),
        (x, y + size),
        (x - size, y),
    ]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, edge, points, 1)


def draw_lamp_icon(surface: pygame.Surface, x: int, y: int, lit: bool = True, size: int = 7) -> None:
    """HUD 用灯泡/光源次数图标。"""
    body = ACCENT_WARM if lit else TEXT_MUTED
    if lit:
        glow = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow, with_alpha(ACCENT_GOLD, 90), (size * 2, size * 2), size + 4)
        surface.blit(glow, (x - size * 2, y - 2 - size * 2))
    pygame.draw.circle(surface, body, (x, y - 2), size)
    pygame.draw.rect(surface, body, (x - 3, y + 3, 6, 5), border_radius=1)
