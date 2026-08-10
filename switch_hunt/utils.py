"""坐标与数值工具函数。"""
from __future__ import annotations

from typing import Tuple
import math

from switch_hunt.constants import TILE_SIZE, MAP_WIDTH, MAP_HEIGHT

def grid_to_pixel(grid_x: int, grid_y: int) -> Tuple[int, int]:
    """
    将网格坐标转换为像素坐标（格子中心点）

    参数:
        grid_x: 网格X坐标
        grid_y: 网格Y坐标

    返回:
        (pixel_x, pixel_y): 像素坐标（格子中心）
    """
    pixel_x = grid_x * TILE_SIZE + TILE_SIZE // 2
    pixel_y = grid_y * TILE_SIZE + TILE_SIZE // 2
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
    grid_x = int(pixel_x // TILE_SIZE)
    grid_y = int(pixel_y // TILE_SIZE)
    grid_x = max(0, min(grid_x, MAP_WIDTH - 1))
    grid_y = max(0, min(grid_y, MAP_HEIGHT - 1))
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
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


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
    return max(min_val, min(value, max_val))
