# 开关猎杀 (Switch Hunt) — 可玩版

Pygame 迷宫捉迷藏：玩家躲藏、开灯定身，鬼由已训练 DQN 推理追击。

> 本分支 `game-only` 仅保留游玩所需代码，已移除训练系统。

## 结构

```
switch_hunt/
  config/     # 光源、鬼速度、出生距离
  core/       # 地图、寻路、实体、视野
  game/       # 音效、UI、主循环
  rl/         # DQN 网络 + 推理加载（不含训练）
apps/play.py  # 启动入口
models/       # ghost_v8.pth（若仓库未含模型，需自行放入）
```

## 安装与运行

```bash
pip install -r requirements.txt
python apps/play.py
# 或
python switch_hunt_v8_game.py
```

将 `models/ghost_v8.pth` 放在项目 `models/` 目录下；缺失时鬼会退化为随机移动。

## 操作

| 按键 | 功能 |
|------|------|
| WASD / 方向键 | 移动 |
| 空格 | 强化光源 |
| F1 | 作弊全图 |
| F2 | 玩家 AI 演示 |
| F3 | 显示鬼 A* 路径 |
| F4 | 音效 |
| P | 暂停 |
| ESC | 菜单 |
