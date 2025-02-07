# train.py

import os
from matplotlib import pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import random_split, DataLoader
import random
import numpy as np
from datetime import datetime

from dataset import PIVDataset
from Dual_CNN_Transformer import TwoStreamPIVNet
from trainer import PIVTrainer
from config import config

def set_random_seed(seed: int = 42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # 1. 设置随机种子
    set_random_seed(42)

    # 2. 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('experiments', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    config['training']['checkpoint_dir'] = os.path.join(save_dir, 'checkpoints')

    # 3. 创建数据集 (单箭头), dataset返回:
    #    frame_a, frame_b => [1,32,32]
    #    velocity => [2,1,1]
    print("\n创建数据集(单箭头)...")
    dataset = PIVDataset(
        root_dir=config['data']['root_dir'],
        phase_range=(1, 60)  # or whatever range
    )

    # 4. 划分训练集和验证集
    train_size = int(len(dataset) * config['data']['train_val_split'])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 5. 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    # 6. 创建模型(单箭头), => [B,2,1,1]
    print("\n创建模型(单箭头)...")
    model = TwoStreamPIVNet(
        base_channels=config['model']['base_channels']
    ).to(config['training']['device'])

    # 7. 创建训练器
    trainer = PIVTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['training']['device'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-4),
        checkpoint_dir=config['training']['checkpoint_dir']
    )

    # 8. 开始训练
    print("\n开始训练(单箭头)...")
    try:
        trainer.train(config['training']['n_epochs'])
    except KeyboardInterrupt:
        print("\n训练被手动中断.")

    # 9. 保存训练历史
    trainer.plot_history()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))

    # 10. 测试最佳模型
    print("\n加载最佳模型进行测试...")
    trainer.load_checkpoint('best_model.pt')
    final_val_loss = trainer.validate()
    print(f"最佳模型验证损失: {final_val_loss:.6f}")

    # 11. (可选) 单箭头可视化:
    #     不再画 32x32 quiver, 而只画1个(u,v)
    print("\n单箭头可视化...")
    save_visualization(model, val_loader, config['training']['device'], save_dir)

def save_visualization(model, val_loader, device, save_dir):
    """
    """
    import os
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    batch = next(iter(val_loader))
    frame_a = batch['frame_a'].to(device)       # [B,1,32,32]
    frame_b = batch['frame_b'].to(device)       # [B,1,32,32]
    velocity_gt = batch['velocity'].to(device)  # [B,2,1,1]  => 单箭头

    with torch.no_grad():
        velocity_pred = model(frame_a, frame_b) # => [B,2,1,1]
        threshold = 0.01  # 可自行调节
        mag = velocity_pred.norm(dim=1, p=2, keepdim=True)  # [B,1,1,1]
        velocity_pred = torch.where(
            mag < threshold,
            torch.zeros_like(velocity_pred),
            velocity_pred
        )

    num_samples = min(4, frame_a.size(0))
    for i in range(num_samples):
        fig, axes = plt.subplots(2,2, figsize=(10,10))

        # 1) 转成 numpy
        img_a = frame_a[i,0].cpu().numpy()  # => [32,32]
        img_b = frame_b[i,0].cpu().numpy()  # => [32,32]
        uv_gt = velocity_gt[i,:,0,0].cpu().numpy()   # => (2,) => (u_gt,v_gt)
        uv_pd = velocity_pred[i,:,0,0].cpu().numpy() # => (2,) => (u_pd,v_pd)

        # =========== 上行 =============
        # (0,0) => Frame A
        axes[0,0].imshow(img_a, cmap='gray', origin='upper')
        axes[0,0].axis('off')
        axes[0,0].set_title(f"Frame A (Sample {i})")

        # (0,1) => Frame B
        axes[0,1].imshow(img_b, cmap='gray', origin='upper')
        axes[0,1].axis('off')
        axes[0,1].set_title("Frame B")

        # =========== 下行 =============
        # (1,0) => Ground Truth Velocity (单箭头)
        # Ground Truth Velocity
        axes[1, 0].set_title("Ground Truth Velocity")
        axes[1, 0].axhline(0, color='k', linewidth=0.5)
        axes[1, 0].axvline(0, color='k', linewidth=0.5)

        # 关键：加入 angles='xy'、scale_units='xy'、scale=1，并设置坐标范围
        axes[1, 0].quiver(
            0, 0,  # 箭头起点 (x0, y0)
            uv_gt[0], uv_gt[1],  # 箭头向量 (u, v)
            angles='xy',
            scale_units='xy',
            scale=1,
            color='green',
            width=0.002,  # 线的宽度，可视情况增减
            headwidth=5,  # 箭头头部宽度
            headlength=7,  # 箭头头部长度
            headaxislength=5,  # 箭头头部沿轴线的长度
            pivot='tail'
        )
        axes[1, 0].set_aspect('equal', adjustable='box')
        axes[1, 0].set_xlim(-0.2, 0.2)
        axes[1, 0].set_ylim(-0.2, 0.2)

        # Predicted Velocity
        axes[1, 1].set_title("Predicted Velocity")
        axes[1, 1].axhline(0, color='k', linewidth=0.5)
        axes[1, 1].axvline(0, color='k', linewidth=0.5)

        axes[1, 1].quiver(
            0, 0,
            uv_pd[0], uv_pd[1],
            angles='xy',
            scale_units='xy',
            scale=1,
            color='red',
            width=0.002,
            headwidth=5,
            headlength=7,
            headaxislength=5,
            pivot='tail'
        )
        axes[1, 1].set_aspect('equal', adjustable='box')
        axes[1, 1].set_xlim(-0.2, 0.2)
        axes[1, 1].set_ylim(-0.2, 0.2)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i}.png'))
        plt.close()



if __name__ == '__main__':
    main()
