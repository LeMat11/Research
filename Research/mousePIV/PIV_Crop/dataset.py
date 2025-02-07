import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from torchvision import transforms

class PIVDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        phase_range: Tuple[int,int] = (1,3),
        transform = None,
        is_training: bool = True
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training

        # 假设图像是32×32
        self.image_size = (32,32)

        print("\n=== 初始化数据集 (单箭头) ===")
        print(f"根目录: {root_dir}")
        print(f"Phase范围: {phase_range}")

        tif_dir = os.path.join(root_dir, 'tif_v1')
        dat_dir = os.path.join(root_dir, 'dat_v2')

        print(f"\n检查目录...")
        print(f"TIF目录 {'存在' if os.path.exists(tif_dir) else '不存在'}: {tif_dir}")
        print(f"DAT目录 {'存在' if os.path.exists(dat_dir) else '不存在'}: {dat_dir}")

        self.samples = []
        total_files = 0
        skipped_files = 0
        size_mismatch = 0

        for phase in range(phase_range[0], phase_range[1]+1):
            phase_str = f"Phase{phase:02d}"

            if not os.path.exists(tif_dir):
                print(f"警告: TIF目录不存在, 跳过 {phase_str}")
                continue

            tif_files = [
                f for f in os.listdir(tif_dir)
                if f.startswith(phase_str) and f.endswith('_A.tif')
            ]
            if not tif_files:
                print(f"在 {phase_str} 中没有找到 TIF文件")
                continue

            print(f"\n处理 {phase_str}...")
            print(f"找到 {len(tif_files)} 个 TIF-A 文件")

            for file in tif_files:
                total_files += 1
                base_name = file[:-6]  # 去 '_A.tif'
                frame_a = os.path.join(tif_dir, f"{base_name}_A.tif")
                frame_b = os.path.join(tif_dir, f"{base_name}_B.tif")
                vel_file= os.path.join(dat_dir, f"{base_name}.npy")

                if not os.path.exists(frame_a):
                    print(f"找不到A帧: {frame_a}")
                    skipped_files += 1
                    continue
                if not os.path.exists(frame_b):
                    print(f"找不到B帧: {frame_b}")
                    skipped_files += 1
                    continue
                if not os.path.exists(vel_file):
                    print(f"找不到速度场文件: {vel_file}")
                    skipped_files += 1
                    continue

                # 检查图像尺寸
                try:
                    with Image.open(frame_a) as img:
                        size = img.size
                        if size != self.image_size:
                            size_mismatch += 1
                            if total_files <= 5:
                                print(f"图像尺寸不匹配 - 期望: {self.image_size}, "
                                      f"实际: {size}, 文件: {frame_a}")
                            continue
                        # 若OK => 加入samples
                        self.samples.append({
                            'frame_a': frame_a,
                            'frame_b': frame_b,
                            'velocity': vel_file,
                            'id': base_name
                        })
                except Exception as e:
                    print(f"读取图像出错 {frame_a}: {str(e)}")
                    skipped_files += 1

        print("\n=== 数据集统计 ===")
        print(f"总文件数: {total_files}")
        print(f"尺寸不匹配: {size_mismatch}")
        print(f"其他跳过文件: {skipped_files}")
        print(f"有效数据对: {len(self.samples)}")

        if len(self.samples) > 0:
            try:
                self._verify_data_shapes(self.samples[0])
            except Exception as e:
                print(f"验证数据形状时出错: {str(e)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1) 读取图像 => 32×32
        frame_a = self._read_tif(sample['frame_a'])
        frame_b = self._read_tif(sample['frame_b'])
        frame_a = torch.FloatTensor(frame_a).unsqueeze(0)  # => [1,32,32]
        frame_b = torch.FloatTensor(frame_b).unsqueeze(0)  # => [1,32,32]

        # 2) 加载速度场 => (1,1,2) "单箭头"
        velocity_np = np.load(sample['velocity'])  # => shape (1,1,2)?

        if velocity_np.shape == (1,1,2):
            # 取出 (u,v)
            uv = velocity_np[0,0,:]  # => shape (2,)
            # reshape => [2,1,1]
            velocity_np = uv.reshape(2,1,1)
        else:
            raise ValueError(f"期望(1,1,2), 实际 {velocity_np.shape}, 文件: {sample['velocity']}")

        velocity = torch.FloatTensor(velocity_np)  # => [2,1,1]

        # 3) 数据增强(可选)
        if self.transform:
            frame_a = self.transform(frame_a)
            frame_b = self.transform(frame_b)
        if self.transform is None and self.is_training:
            frame_a, frame_b = self._random_augmentation(frame_a, frame_b)

        return {
            'frame_a': frame_a,     # [1,32,32]
            'frame_b': frame_b,     # [1,32,32]
            'velocity': velocity,   # [2,1,1]
            'sample_id': sample['id']
        }

    def _read_tif(self, path: str) -> np.ndarray:
        with Image.open(path) as img:
            data = np.array(img, dtype=np.float32)
            data = (data - data.min()) / (data.max() - data.min())
            return data

    def _verify_data_shapes(self, sample: Dict):
        """可选: 测试一个样本形状"""
        img_a = self._read_tif(sample['frame_a'])
        vel   = np.load(sample['velocity'])
        if img_a.shape != self.image_size:
            raise ValueError(f"图像尺寸应为 {self.image_size}, 实际 {img_a.shape}")
        print(f"图像形状: {img_a.shape}")
        print(f"速度场形状: {vel.shape}")

    def _random_augmentation(self, img1, img2):
        """
        如果要做随机增广, 这里写.
        对单箭头无影响, 只增广图像.
        """
        return img1, img2


def create_dataloader(
    root_dir: str,
    batch_size: int,
    phase_range: Tuple[int,int] = (1,3),
    transform=None,
    is_training: bool = True,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    dataset = PIVDataset(
        root_dir=root_dir,
        phase_range=phase_range,
        transform=transform,
        is_training=is_training
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


if __name__ == "__main__":
    # 测试
    dataset = PIVDataset(
        root_dir='path/to/your/data',
        phase_range=(1,3),
        is_training=True
    )
    sample = dataset[0]
    print(f"Frame A shape : {sample['frame_a'].shape}")   # => [1,32,32]
    print(f"Frame B shape : {sample['frame_b'].shape}")   # => [1,32,32]
    print(f"Velocity shape: {sample['velocity'].shape}")  # => [2,1,1]
    print(f"Sample ID     : {sample['sample_id']}")
