import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrelationLayer(nn.Module):
    def __init__(self, max_displacement=4):
        super().__init__()
        self.max_disp = max_displacement
        self.kernel_size = 2 * max_displacement + 1  # = 9
        self.pad_size   = max_displacement          # = 4

    def forward(self, feat1, feat2):
        """
        feat1, feat2: [B, C, 4, 4]  (三次池化后)
        """
        B, C, H, W = feat1.shape
        feat2_pad = F.pad(feat2, [self.pad_size]*4, mode='constant')  # => [B,C,12,12]

        # 裁剪 9×9 区域 => [B,C,9,9]
        feat2_crop = feat2_pad[:, :, 2:11, 2:11]

        # reshape => [B*C, 1, 9,9]
        weight_ = feat2_crop.view(B*C, 1, self.kernel_size, self.kernel_size)

        # input => [1, B*C, 4,4]
        input_ = feat1.view(1, B*C, H, W)

        # 分组卷积 => output => [1,B*C, 4,4]
        corr = F.conv2d(
            input=input_,
            weight=weight_,
            bias=None,
            stride=1,
            padding=self.pad_size,
            groups=B*C
        )
        corr = corr.view(B, C, H, W)
        # 在通道上做 sum => [B,1,4,4]
        corr = corr.sum(dim=1, keepdim=True)
        return corr

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels,       base_channels,   3, padding=1)
        self.conv2 = nn.Conv2d(base_channels,        base_channels*2, 3, padding=1)
        self.conv3 = nn.Conv2d(base_channels*2,      base_channels*4, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.norm1= nn.BatchNorm2d(base_channels)
        self.norm2= nn.BatchNorm2d(base_channels*2)
        self.norm3= nn.BatchNorm2d(base_channels*4)

    def forward(self, x):
        # => [B, base_channels, 32, 32] after conv1, but we do 3x pool => [B, *, 4,4]
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.norm3(self.conv3(x)))
        x = self.pool(x)  # => [B, base_channels*4, 4,4]
        return x

class FlowEstimator(nn.Module):
    """
    最终只输出 [B,2,1,1] => 单箭头
    做法:
      1) 卷积到 [B,2,4,4]
      2) 全局池化 => [B,2,1,1]
    """
    def __init__(self, input_channels=1, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, base_channels*4, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1)
        self.conv3 = nn.Conv2d(base_channels*2, base_channels,   3, padding=1)
        self.conv4 = nn.Conv2d(base_channels,   2,               3, padding=1)

        self.norm1 = nn.BatchNorm2d(base_channels*4)
        self.norm2 = nn.BatchNorm2d(base_channels*2)
        self.norm3 = nn.BatchNorm2d(base_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        # x => [B,1,4,4]
        x = self.relu(self.norm1(self.conv1(x)))  # => [B,256,4,4] if base=64
        x = self.relu(self.norm2(self.conv2(x)))  # => [B,128,4,4]
        x = self.relu(self.norm3(self.conv3(x)))  # => [B,64,4,4]
        x = self.conv4(x)                         # => [B,2,4,4]

        # 全局池化 => [B,2,1,1]
        x = F.adaptive_avg_pool2d(x, (1,1))
        return x

class TwoStreamPIVNet(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_channels=1, base_channels=base_channels)
        self.correlation = CorrelationLayer(max_displacement=4)
        self.flow_estimator = FlowEstimator(input_channels=1, base_channels=base_channels)
        # 不再做上采样, 直接单箭头 => [B,2,1,1]

    def forward(self, frame1, frame2):
        feat1 = self.feature_extractor(frame1)  # => [B,256,4,4]
        feat2 = self.feature_extractor(frame2)

        corr  = self.correlation(feat1, feat2)  # => [B,1,4,4]

        flow  = self.flow_estimator(corr)       # => [B,2,1,1]
        return flow


if __name__ == '__main__':
    # 测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TwoStreamPIVNet(base_channels=64).to(device)

    batch_size = 2
    frame1 = torch.randn(batch_size, 1, 32, 32).to(device)
    frame2 = torch.randn(batch_size, 1, 32, 32).to(device)

    with torch.no_grad():
        out = model(frame1, frame2)
        print("网络输出 shape:", out.shape)
        # => [2,2,1,1] => batch=2, channel=2, 1×1 => 单箭头
