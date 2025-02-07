import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
config = {
    'data': {
        'root_dir': '../32by32',
        'batch_size': 32,
        'num_workers': 4,
        'train_val_split': 0.8,  # 训练集比例
        'phase_range': (1, 60)  # 统一相位范围
    },
    'model': {
        'input_size': (32, 32),  # 统一输入尺寸
        'base_channels': 64,  # 基础通道数
    },
    'training': {
        'device': 'cuda',
        'n_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,  # 添加权重衰减
        'gradient_clip': 0.1,  # 调整梯度裁剪阈值
        'checkpoint_dir': 'checkpoints',
        'scheduler': {
            'factor': 0.5,  # 学习率衰减因子
            'patience': 5,  # 容忍epochs数
            'min_lr': 1e-6  # 最小学习率
        }
    },
    'loss': {
        'direction_weight': 1,  # 方向损失权重
        'magnitude_weight': 1  # 幅值损失权重
    }
}