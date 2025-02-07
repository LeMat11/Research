# trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from losses import PIVLoss
from config import config

class PIVTrainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: str,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度 (ReduceLROnPlateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['training']['scheduler']['factor'],
            patience=config['training']['scheduler']['patience'],
            min_lr=config['training']['scheduler']['min_lr'],
            verbose=True
        )
        self.criterion = PIVLoss(
            magnitude_weight=config['loss']['magnitude_weight'],
            direction_weight=config['loss']['direction_weight']
        )

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            frame_a = batch['frame_a'].to(self.device)      # [B,1,32,32]
            frame_b = batch['frame_b'].to(self.device)      # [B,1,32,32]
            velocity_gt = batch['velocity'].to(self.device) # [B,2,1,1]

            self.optimizer.zero_grad()

            # 前向传播 => [B,2,1,1]
            velocity_pred = self.model(frame_a, frame_b)

            # 单箭头loss => MSE
            loss = self.criterion(velocity_pred, velocity_gt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config['training']['gradient_clip']
            )
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                frame_a = batch['frame_a'].to(self.device)      # [B,1,32,32]
                frame_b = batch['frame_b'].to(self.device)
                velocity_gt = batch['velocity'].to(self.device) # [B,2,1,1]

                velocity_pred = self.model(frame_a, frame_b)    # => [B,2,1,1]
                loss = self.compute_loss(velocity_pred, velocity_gt)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def compute_loss(self, pred, target):
        """
        对单箭头 [B,2,1,1] 做 MSELoss.
        也可改"幅值+方向"组合损失, 只要形状一致就好.
        """
        return F.mse_loss(pred, target)

    def train(self, n_epochs):
        """主训练入口"""
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            train_loss = self.train_epoch()
            val_loss   = self.validate()

            # 更新学习率
            self.scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch + 1}.pt')

    def save_checkpoint(self, filename):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))

    def load_checkpoint(self, filename):
        """加载检查点"""
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses   = checkpoint['val_losses']
        self.best_val_loss= checkpoint['best_val_loss']

    def plot_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses,   label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Single-Arrow Training History')
        plt.legend()
        plt.grid(True)

    def save_visualization(self, save_dir="visualizations"):
        """
        单箭头可视化(可选):
        - 取 val_loader里一个 batch
        - 对前若干样本, 在图像中心画1支 GT箭头(绿) & 1支 Pred箭头(红)
        """
        import os
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()

        batch = next(iter(self.val_loader))
        frame_a   = batch['frame_a'].to(self.device)      # [B,1,32,32]
        frame_b   = batch['frame_b'].to(self.device)
        velocity_gt = batch['velocity'].to(self.device)   # [B,2,1,1]

        with torch.no_grad():
            velocity_pred = self.model(frame_a, frame_b)  # => [B,2,1,1]

        num_samples = min(4, frame_a.size(0))
        for i in range(num_samples):
            fig, ax = plt.subplots(figsize=(5,5))

            img_a = frame_a[i,0].cpu().numpy()  # => [32,32]
            uv_gt = velocity_gt[i,:,0,0].cpu().numpy()   # => (2,) => (u_gt,v_gt)
            uv_pd = velocity_pred[i,:,0,0].cpu().numpy() # => (2,) => (u_pd,v_pd)

            ax.imshow(img_a, cmap='gray', origin='upper')
            ax.invert_yaxis()
            ax.set_title(f"Sample {i}: single-arrow")

            # 画GT箭头(绿) & Pred箭头(红)在中心(16,16)
            x0, y0 = 16, 16
            ax.quiver(x0, y0, uv_gt[0], uv_gt[1], color='green', angles='xy', scale_units='xy', scale=1, label='GT')
            ax.quiver(x0, y0, uv_pd[0], uv_pd[1], color='red',   angles='xy', scale_units='xy', scale=1, label='Pred')

            ax.set_xlim([0,32])
            ax.set_ylim([0,32])
            ax.legend()

            plt.savefig(os.path.join(save_dir, f'sample_{i}.png'))
            plt.close()
