import torch
import torch.nn as nn
import torch.nn.functional as F

class PIVLoss(nn.Module):
    def __init__(self, magnitude_weight=1.0, direction_weight=1.0):
        """
        magnitude_weight: 幅值损失(MSE)的权重
        direction_weight: 方向损失(1 - cos_sim)的权重
        """
        super().__init__()
        self.magnitude_weight = magnitude_weight
        self.direction_weight = direction_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        """
        pred, target: [B, 2, 1, 1] (单箭头)
        """
        # 1) 幅值损失(MSE)
        loss_data = self.mse(pred, target)

        # 2) (可选) 平滑项 => 单箭头无空间维度, 返回0, 不参与总损失
        loss_smooth = self._smoothness_loss(pred)

        # 3) 方向一致性损失(1 - cos_sim)
        loss_direction = self._direction_loss(pred, target)

        # 最终Loss = 幅值损失 * (magnitude_weight) + 方向损失 * (direction_weight)
        # 平滑项对单箭头无意义, 不加入
        return (self.magnitude_weight * loss_data
                + self.direction_weight * loss_direction)

    def _smoothness_loss(self, velocity):
        """
        单箭头模式: [B,2,1,1], 无空间维度可做梯度/拉普拉斯 => 返回0
        """
        return torch.tensor(0.0, device=velocity.device)

    def _direction_loss(self, pred, target):
        """
        方向损失:
          dot_product = (pred * target).sum(dim=1) => [B, *]
          cos_sim = dot / (||pred|| * ||target|| + 1e-6)
        结果越接近1，方向越一致；因此损失定义为 1 - cos_sim 的平均。
        """
        dot_product = (pred * target).sum(dim=1)
        norm_pred   = torch.norm(pred, dim=1)
        norm_target = torch.norm(target, dim=1)
        cos_sim     = dot_product / (norm_pred * norm_target + 1e-6)

        return 1 - cos_sim.mean()
