# Dual_CNN_Transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------- 可选: 使用 einops(更简洁) -------------
# 如果没有 einops，可以注释掉这块，并使用下面的 "无 einops" 写法
try:
    from einops import rearrange
    USE_EINOPS = True
except ImportError:
    USE_EINOPS = False


class TransformerBlock(nn.Module):
    """
    一个小型Transformer编码器示例:
      d_model=128, nhead=4, dim_feedforward=256, num_layers=1
    可根据需求自行调整超参数.
    """
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # PyTorch 1.9+
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [B, C, H, W]
        需要先把空间维度(H,W)展平成序列，然后过Transformer，再reshape回来.
        """
        B, C, H, W = x.shape

        if USE_EINOPS:
            # 用 einops 重排 (更简洁)
            # 1) -> [B, H*W, C]
            x_reshaped = rearrange(x, 'b c h w -> b (h w) c')
            # 2) Transformer
            x_trans = self.transformer(x_reshaped)  # 仍是 [B, (H*W), C]
            # 3) -> [B, C, H, W]
            x_out = rearrange(x_trans, 'b (h w) c -> b c h w', h=H, w=W)
        else:
            # 无 einops 写法
            # 1) permute => [B, C, H, W] -> [B, H, W, C]
            x_permuted = x.permute(0, 2, 3, 1).contiguous()
            # 2) reshape => [B, H, W, C] -> [B, H*W, C]
            x_reshaped = x_permuted.view(B, H*W, C)
            # 3) Transformer => [B, H*W, C]
            x_trans = self.transformer(x_reshaped)
            # 4) reshape -> [B, H, W, C]
            x_trans = x_trans.view(B, H, W, C)
            # 5) permute => [B, C, H, W]
            x_out = x_trans.permute(0, 3, 1, 2).contiguous()

        return x_out


class FeatureExtractor(nn.Module):
    """
    在原先的三层卷积+池化结构中,
    在中间插入一个 TransformerBlock, 从而让网络捕捉全局注意力.
    """
    def __init__(self, input_channels=1, base_channels=64):
        super().__init__()
        # block1
        self.conv1 = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels,   base_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(base_channels)
        self.pool1 = nn.MaxPool2d(2)  # 32->16

        # block2
        self.conv3 = nn.Conv2d(base_channels, base_channels*2, 3, padding=1)
        self.conv4 = nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(base_channels*2)
        # 这里不急着pool, 先插入Transformer
        # 通过pooling之前的特征图做自注意力, 维度更大,更灵活
        self.transformer_block = TransformerBlock(
            d_model=base_channels*2,  # 通道数=128 if base_channels=64
            nhead=4,
            dim_feedforward=256,
            num_layers=1
        )
        self.pool2 = nn.MaxPool2d(2)  # 16->8

        # block3
        self.conv5 = nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1)
        self.conv6 = nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(base_channels*4)
        self.pool3 = nn.MaxPool2d(2)  # 8->4

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # block1
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.pool1(x)  # => [B, base_channels, 16,16]

        # block2
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.bn2(x)
        # 现在 x => [B, base_channels*2, 16,16]

        # 引入 TransformerBlock 做自注意力
        x = self.transformer_block(x)
        # => [B, base_channels*2, 16,16]

        x = self.pool2(x)  # => [B, base_channels*2, 8,8]

        # block3
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.bn3(x)
        x = self.pool3(x)  # => [B, base_channels*4, 4,4]
        return x


class CorrelationLayer(nn.Module):
    """
    与之前相同的相关运算(FlowNet式), 在 4x4 特征图上做局部卷积.
    """
    def __init__(self, max_displacement=4):
        super().__init__()
        self.max_disp = max_displacement
        self.kernel_size = 2 * max_displacement + 1  # = 9
        self.pad_size = max_displacement

    def forward(self, feat1, feat2):
        """
        feat1, feat2: [B, C, 4, 4]  (三次池化后)
        输出: [B, 1, 4, 4], 再与FlowEstimator衔接.
        """
        B, C, H, W = feat1.shape
        feat2_pad = F.pad(feat2, [self.pad_size]*4, mode='constant')  # => [B,C,12,12]

        # 简化: 只取中心9x9进行卷积
        feat2_crop = feat2_pad[:, :, 2:11, 2:11]  # => [B,C,9,9]

        # reshape => [B*C,1,9,9]
        weight_ = feat2_crop.view(B*C, 1, self.kernel_size, self.kernel_size)

        # input => [1, B*C, 4,4]
        input_ = feat1.view(1, B*C, H, W)

        corr = F.conv2d(
            input=input_,
            weight=weight_,
            bias=None,
            stride=1,
            padding=self.pad_size,
            groups=B*C
        )
        corr = corr.view(B, C, H, W)
        corr = corr.sum(dim=1, keepdim=True)  # 在通道上求和 => [B,1,4,4]
        return corr


class FlowEstimator(nn.Module):
    """
    根据 [B,1,4,4] 的 cost map 估计单箭头 => [B,2,1,1].
    """
    def __init__(self, input_channels=1, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, base_channels*4, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1)
        self.conv3 = nn.Conv2d(base_channels*2, base_channels,   3, padding=1)
        self.conv4 = nn.Conv2d(base_channels,   2,               3, padding=1)

        self.bn1 = nn.BatchNorm2d(base_channels*4)
        self.bn2 = nn.BatchNorm2d(base_channels*2)
        self.bn3 = nn.BatchNorm2d(base_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # => [B,256,4,4] if base=64
        x = self.relu(self.bn2(self.conv2(x)))  # => [B,128,4,4]
        x = self.relu(self.bn3(self.conv3(x)))  # => [B,64,4,4]
        x = self.conv4(x)                       # => [B,2,4,4]

        # 全局池化 => [B,2,1,1]
        x = F.adaptive_avg_pool2d(x, (1,1))
        return x


class TwoStreamPIVNet(nn.Module):
    """
    包含:
      1) 特征提取(带自注意力)
      2) CorrelationLayer
      3) FlowEstimator
    仍输出单箭头: [B,2,1,1]
    """
    def __init__(self, base_channels=64):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_channels=1, base_channels=base_channels)
        self.correlation = CorrelationLayer(max_displacement=4)
        self.flow_estimator = FlowEstimator(input_channels=1, base_channels=base_channels)

    def forward(self, frame1, frame2):
        # 1) 提取特征
        feat1 = self.feature_extractor(frame1)  # [B, base_channels*4, 4,4]
        feat2 = self.feature_extractor(frame2)
        # 2) 做互相关(或cost volume)
        corr  = self.correlation(feat1, feat2)   # [B,1,4,4]
        # 3) flow估计 => [B,2,1,1]
        flow  = self.flow_estimator(corr)
        return flow


if __name__ == '__main__':
    """
    简单测试: 构造随机输入, 看输出尺寸是否正确.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoStreamPIVNet(base_channels=64).to(device)

    # 模拟一个batch_size=2, 1通道, 32x32图像
    frame1 = torch.randn(2, 1, 32, 32, device=device)
    frame2 = torch.randn(2, 1, 32, 32, device=device)

    with torch.no_grad():
        out = model(frame1, frame2)
    print("Output shape:", out.shape)  # => [2,2,1,1]
    print("Done.")
