"""
DQN V8 快速训练脚本
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description='DQN V8.0 - Grid-Aligned Direction Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py                    # 默认训练500回合
  python train.py --episodes 1000    # 训练1000回合
  python train.py --render 1         # 带实时画面训练
  python train.py -e 100 -r 1 -p 5   # 100回合，渲染，每5回合打印
        """
    )
    
    parser.add_argument('-e', '--episodes', type=int, default=500,
                        help='训练回合数 (默认: 500)')
    parser.add_argument('-p', '--print-every', type=int, default=10,
                        help='打印频率 (默认: 10)')
    parser.add_argument('-r', '--render', type=int, default=0, choices=[0, 1],
                        help='实时渲染开关 (1=开, 0=关, 默认: 0)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DQN V8.0 - Grid-Aligned Direction Learning")
    print("=" * 70)
    print(f"\n训练参数:")
    print(f"  回合数: {args.episodes}")
    print(f"  渲染: {'开启' if args.render else '关闭'}")
    print(f"  打印间隔: 每 {args.print_every} 回合")
    print()
    
    if args.render:
        print("渲染模式提示:")
        print("  - 按 ESC 停止训练")
        print("  - 关闭窗口停止训练")
        print()
    
    print("=" * 70)
    print()
    
    # 导入并启动训练
    from dqn_training_system_v8 import train_ghost_v8
    
    try:
        train_ghost_v8(
            episodes=args.episodes,
            print_every=args.print_every,
            render=args.render == 1
        )
    except KeyboardInterrupt:
        print("\n\n[用户中断] 训练已停止")
        sys.exit(0)

if __name__ == "__main__":
    main()
