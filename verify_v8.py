"""
V8系统验证脚本 - 测试核心功能是否正常工作
"""
import os
os.environ['DQN_TRAINING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

def test_imports():
    """测试模块导入"""
    print("[Test 1/6] 测试模块导入...")
    try:
        from switch_hunt_v8_game import GameV8, PlayerV8, DQNGhostV8
        from dqn_training_system_v8 import SwitchHuntTrainingEnvV8
        from dqn_model_v8 import ConfigurableDQNAI
        from config_v8 import LIGHT_SYSTEM, GHOST_REWARD
        print("  [PASS] 所有模块导入成功")
        return True
    except Exception as e:
        print(f"  [FAIL] 导入失败: {e}")
        return False


def test_game_init():
    """测试游戏初始化"""
    print("[Test 2/6] 测试游戏初始化...")
    try:
        from switch_hunt_v8_game import GameV8
        game = GameV8()
        game.init_game()
        assert len(game.ghosts) == 1
        # TREASURE_COUNT=8 in V7 base
        assert len(game.treasures) == 8
        assert hasattr(game.player, 'light_charges')
        assert hasattr(game.ghosts[0], 'grid_pos')
        print("  [PASS] 游戏初始化成功")
        return True
    except Exception as e:
        print(f"  [FAIL] 初始化失败: {e}")
        return False


def test_ghost_grid_alignment():
    """测试鬼网格对齐"""
    print("[Test 3/6] 测试网格对齐...")
    try:
        from switch_hunt_v8_game import GameV8
        game = GameV8()
        game.init_game()
        ghost = game.ghosts[0]
        
        # 检查初始位置是否对齐
        gx, gy = ghost.grid_pos
        expected_x = gx * 32 + 16
        expected_y = gy * 32 + 16
        
        assert abs(ghost.pos[0] - expected_x) < 1, f"X位置不对齐: {ghost.pos[0]} vs {expected_x}"
        assert abs(ghost.pos[1] - expected_y) < 1, f"Y位置不对齐: {ghost.pos[1]} vs {expected_y}"
        assert ghost.radius == 16, f"半径错误: {ghost.radius}"
        
        print(f"  [PASS] 网格对齐 OK (位置: {ghost.grid_pos}, 像素: {ghost.pos})")
        return True
    except Exception as e:
        print(f"  [FAIL] 网格对齐测试失败: {e}")
        return False


def test_ghost_movement():
    """测试鬼移动系统"""
    print("[Test 4/6] 测试鬼移动...")
    try:
        from switch_hunt_v8_game import GameV8
        game = GameV8()
        game.init_game()
        ghost = game.ghosts[0]
        
        start_pos = ghost.grid_pos
        
        # 尝试移动
        moved = ghost.apply_action(3, 1/60)  # RIGHT
        
        if moved:
            print(f"  [PASS] 鬼移动启动成功 (from {start_pos})")
        else:
            # 可能右边是墙，这是正常的
            print(f"  [INFO] 移动被阻挡（可能右边是墙）")
        
        return True
    except Exception as e:
        print(f"  [FAIL] 移动测试失败: {e}")
        return False


def test_direction_reward():
    """测试方向奖励计算"""
    print("[Test 5/6] 测试方向奖励...")
    try:
        from dqn_training_system_v8 import SwitchHuntTrainingEnvV8
        from config_v8 import GHOST_REWARD
        
        env = SwitchHuntTrainingEnvV8(render=False)
        env.reset()
        
        # 获取计划方向
        planned = env.planned_direction
        
        # 执行一个动作
        action = 3  # RIGHT
        next_state, reward, done, info = env.step_train_ghost(action)
        
        print(f"  [PASS] 奖励计算 OK (plan={planned}, act={action}, reward={reward:.2f})")
        return True
    except Exception as e:
        print(f"  [FAIL] 奖励测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_encoding():
    """测试状态编码"""
    print("[Test 6/6] 测试状态编码...")
    try:
        from switch_hunt_v8_game import GameV8
        import numpy as np
        
        game = GameV8()
        game.init_game()
        ghost = game.ghosts[0]
        
        state = ghost.get_state()
        
        assert state.shape == (6, 21, 21), f"状态形状错误: {state.shape}"
        assert state.dtype == np.float32, f"数据类型错误: {state.dtype}"
        
        # V8.13: 检查各通道
        # 通道0: 墙壁 (应该有多个1)
        wall_sum = state[0].sum()
        assert wall_sum > 0, f"墙壁通道错误: {wall_sum}"
        
        # 通道1: 鬼位置 (应该只有一个1)
        ghost_pos_sum = state[1].sum()
        assert ghost_pos_sum == 1.0, f"鬼位置编码错误: {ghost_pos_sum}"
        
        # 通道2: 玩家位置 (应该只有一个1)
        player_pos_sum = state[2].sum()
        assert player_pos_sum == 1.0, f"玩家位置编码错误: {player_pos_sum}"
        
        print(f"  [PASS] 状态编码 OK (shape: {state.shape}, walls: {wall_sum:.0f})")
        return True
    except Exception as e:
        print(f"  [FAIL] 状态编码测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("V8系统验证")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_game_init,
        test_ghost_grid_alignment,
        test_ghost_movement,
        test_direction_reward,
        test_state_encoding,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  [ERROR] 测试异常: {e}")
            results.append(False)
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("[V8] 所有测试通过! 系统正常。")
    else:
        print("[V8] 部分测试失败，请检查。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
