import torch
import os

models_dir = 'D:/03-学习资料/AI学习/硕士/Term2_ML/individual_asg1/重要/三个模型'
models = ['ghost_v8.pth', 'ghost_v85002.pth', 'ghost_v85003p.pth']

for model_name in models:
    path = os.path.join(models_dir, model_name)
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu')
        print(f'=== {model_name} ===')
        print(f'  File size: {os.path.getsize(path) / 1024 / 1024:.2f} MB')
        print(f'  Epsilon: {checkpoint.get("epsilon", "N/A")}')
        print(f'  Step count: {checkpoint.get("step_count", "N/A")}')
        # Check conv1 channels
        if 'policy_net' in checkpoint:
            policy = checkpoint['policy_net']
            conv1_key = None
            for k in policy.keys():
                if 'conv1.weight' in k:
                    conv1_key = k
                    break
            if conv1_key:
                print(f'  Input channels: {policy[conv1_key].shape[1]}')
        print()
