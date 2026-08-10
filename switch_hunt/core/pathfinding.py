"""A* 寻路。"""
from __future__ import annotations

import heapq
from typing import List, Tuple, Optional

class AStarPathfinder:
    """A*寻路算法实现类"""

    def __init__(self, map_obj: Map):
        """
        初始化A*寻路器

        参数:
            map_obj: 地图对象
        """
        self.map = map_obj

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        启发函数：使用曼哈顿距离

        参数:
            a: 起点格子坐标
            b: 终点格子坐标

        返回:
            估计距离
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A*寻路主函数

        参数:
            start: 起点格子坐标
            goal: 目标格子坐标

        返回:
            路径列表，每个元素是格子坐标
        """
        # 如果起点或终点是墙壁，返回空路径
        if self.map.is_wall(start[0], start[1]) or self.map.is_wall(goal[0], goal[1]):
            return []

        if start == goal:
            return [start]

        # 初始化开放列表（优先队列）
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start))

        # 记录每个节点的来源
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # g_score: 从起点到当前节点的实际代价
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        # f_score: 估计的总代价
        f_score: Dict[Tuple[int, int], int] = {start: self.heuristic(start, goal)}

        # 开放列表中的节点集合
        open_set_hash = {start}

        # 四个移动方向
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while open_set:
            # 取出f_score最小的节点
            current_f, _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            # 到达目标，重建路径
            if current == goal:
                return self._reconstruct_path(came_from, current)

            # 遍历邻居
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                # 检查邻居是否可行走
                if not self.map.is_wall(neighbor[0], neighbor[1]):
                    tentative_g = g_score[current] + 1

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)

                        if neighbor not in open_set_hash:
                            counter += 1
                            heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                            open_set_hash.add(neighbor)

        # 没有找到路径
        return []

    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        重建路径

        参数:
            came_from: 记录每个节点来源的字典
            current: 终点节点

        返回:
            从起点到终点的路径列表
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# =============================================================================
# 第六部分：玩家类
# =============================================================================
