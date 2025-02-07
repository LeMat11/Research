import os
import numpy as np
from PIL import Image
from typing import Dict, Tuple


class PIVProcessor:
    def __init__(
            self,
            dat_dir: str = r'\\10.24.62.51\Lab Shared\Project Mouse PIV\PreDataSet\Average',
            tif_dir: str = r'\\10.24.62.51\Lab Shared\Project Mouse PIV\PreDataSet\DoubleFrameAverage',
            save_dir: str = r'\\10.24.62.51\Lab Shared\Project Mouse PIV\PreDataSet\30by30',
            tif_size: int = 32
    ):
        self.dat_dir = dat_dir
        self.tif_dir = tif_dir
        self.save_dir = save_dir
        self.tif_size = tif_size

        # 创建保存目录
        os.makedirs(os.path.join(save_dir, 'tif_v1'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'dat_v2'), exist_ok=True)

    def get_velocity_block_size(self, tif_shape: Tuple[int, int], vel_shape: Tuple[int, int]) -> Tuple[int, int]:
        """计算速度场块的正确尺寸"""
        tif_h, tif_w = tif_shape
        vel_h, vel_w = vel_shape[:2]

        # 计算实际的比例
        h_ratio = tif_h / vel_h
        w_ratio = tif_w / vel_w

        print(f"图像与速度场的比例 - 高度: {h_ratio:.2f}, 宽度: {w_ratio:.2f}")

        # 计算对应于tif_size的速度场块大小
        vel_block_h = int(round(self.tif_size / h_ratio))
        vel_block_w = int(round(self.tif_size / w_ratio))

        return vel_block_h, vel_block_w

    def process_single_phase(self, phase_num: int) -> Dict:
        """处理单个Phase的数据"""
        phase = f"Phase{phase_num:02d}"

        # 构建文件路径
        dat_path = os.path.join(self.dat_dir, f"{phase}_Averaged.dat")
        tif1_path = os.path.join(self.tif_dir, f"{phase}FrameA.tif")
        tif2_path = os.path.join(self.tif_dir, f"{phase}FrameB.tif")

        print(f"\n正在处理 {phase}...")
        print(f"dat文件: {dat_path}")
        print(f"tif文件A: {tif1_path}")
        print(f"tif文件B: {tif2_path}")

        # 读取数据
        img1 = self._read_tif(tif1_path)
        img2 = self._read_tif(tif2_path)
        vel_field = self._read_dat(dat_path)

        print(f"图像尺寸: {img1.shape}")
        print(f"速度场尺寸: {vel_field.shape}")

        # 计算速度场块的尺寸
        vel_block_h, vel_block_w = self.get_velocity_block_size(img1.shape, vel_field.shape)
        print(f"速度场块的目标尺寸: {vel_block_h}x{vel_block_w}")

        # 切割和保存
        h, w = img1.shape
        n_blocks_h = h // self.tif_size
        n_blocks_w = w // self.tif_size

        print(f"将切割成 {n_blocks_h}×{n_blocks_w} 个块")
        blocks_info = []

        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                # 图像块范围
                h_start = i * self.tif_size
                h_end = h_start + self.tif_size
                w_start = j * self.tif_size
                w_end = w_start + self.tif_size

                # 速度场块范围
                vh_start = i * vel_block_h
                vh_end = vh_start + vel_block_h
                vw_start = j * vel_block_w
                vw_end = vw_start + vel_block_w

                # 提取数据块
                img1_block = img1[h_start:h_end, w_start:w_end]
                img2_block = img2[h_start:h_end, w_start:w_end]

                # 检查是否超出范围
                if vh_end <= vel_field.shape[0] and vw_end <= vel_field.shape[1]:
                    vel_block = vel_field[vh_start:vh_end, vw_start:vw_end]

                    # 检查块的完整性
                    if (img1_block.shape == (self.tif_size, self.tif_size) and
                            vel_block.shape == (vel_block_h, vel_block_w, 2)):

                        # 生成块ID并保存
                        block_id = f"{phase}_{i:03d}_{j:03d}"
                        self._save_block(block_id, img1_block, img2_block, vel_block)

                        if i == 0 and j == 0:  # 打印第一个块的信息
                            print(f"\n第一个数据块信息:")
                            print(f"图像块尺寸: {img1_block.shape}")
                            print(f"速度场块尺寸: {vel_block.shape}")

                            # 保存可视化样本
                            self._save_sample_visualization(img1_block, img2_block, vel_block, phase)

                        blocks_info.append({
                            'block_id': block_id,
                            'image_range': (h_start, h_end, w_start, w_end),
                            'velocity_range': (vh_start, vh_end, vw_start, vw_end)
                        })

        print(f"\n处理完成!")
        print(f"总数据块数: {len(blocks_info)}")

        return {
            'phase_id': phase_num,
            'tif1_path': tif1_path,
            'tif2_path': tif2_path,
            'dat_path': dat_path,
            'n_blocks': len(blocks_info),
            'blocks_info': blocks_info
        }

    def _read_tif(self, path: str) -> np.ndarray:
        """读取tif图像"""
        with Image.open(path) as img:
            return np.array(img)

    def _read_dat(self, path: str) -> np.ndarray:
        """读取dat速度场文件"""
        data = np.loadtxt(path, skiprows=5)

        # 确定网格尺寸
        x_coords = data[:, 0]
        y_coords = data[:, 1]

        # 找到唯一的x和y坐标值
        unique_x = np.unique(x_coords)
        unique_y = np.unique(y_coords)

        w = len(unique_x)
        h = len(unique_y)

        # 初始化速度场数组
        vel_field = np.zeros((h, w, 2))

        # 创建坐标到索引的映射
        x_to_idx = {x: i for i, x in enumerate(unique_x)}
        y_to_idx = {y: i for i, y in enumerate(unique_y)}

        # 填充速度场数据
        for i in range(len(data)):
            x, y = data[i, 0], data[i, 1]
            col = x_to_idx[x]
            row = y_to_idx[y]
            vel_field[row, col, 0] = data[i, 2]  # u分量
            vel_field[row, col, 1] = data[i, 3]  # v分量

        return vel_field

    def _save_block(self, block_id: str, img1: np.ndarray, img2: np.ndarray, vel: np.ndarray):
        """保存数据块"""
        # 保存tif文件
        tif1_path = os.path.join(self.save_dir, 'tif_v1', f'{block_id}_A.tif')
        tif2_path = os.path.join(self.save_dir, 'tif_v1', f'{block_id}_B.tif')
        Image.fromarray(img1).save(tif1_path)
        Image.fromarray(img2).save(tif2_path)

        # 保存速度场数据
        dat_path = os.path.join(self.save_dir, 'dat_v2', f'{block_id}.npy')
        np.save(dat_path, vel)

    def _save_sample_visualization(self, img1, img2, vel, phase):
        """保存样本可视化，处理零速度和极小值"""
        import matplotlib.pyplot as plt

        # 创建sample目录
        sample_dir = os.path.join(self.save_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)

        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 显示第一帧
        axes[0].imshow(img1, cmap='gray')
        axes[0].set_title('Frame A')
        axes[0].axis('off')

        # 显示第二帧
        axes[1].imshow(img2, cmap='gray')
        axes[1].set_title('Frame B')
        axes[1].axis('off')

        # 显示速度场
        x = np.arange(vel.shape[1])
        y = np.arange(vel.shape[0])
        X, Y = np.meshgrid(x, y)

        # 计算速度幅值
        vel_magnitude = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2)

        # 设置最小阈值，避免除零
        min_magnitude = np.max(vel_magnitude) * 1e-3  # 使用最大值的0.1%作为阈值
        mask = vel_magnitude > min_magnitude

        if np.any(mask):  # 如果有有效的速度向量
            axes[2].quiver(X[mask], Y[mask],
                           vel[mask, 0], vel[mask, 1],
                           vel_magnitude[mask],
                           angles='xy', scale_units='xy', scale=1,
                           cmap='jet')
            axes[2].set_title('Velocity Field')
        else:
            axes[2].text(0.5, 0.5, 'No significant velocity',
                         ha='center', va='center')

        axes[2].set_aspect('equal')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, f'{phase}_sample.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 额外保存速度场数据的统计信息
        stats_file = os.path.join(sample_dir, f'{phase}_vel_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Velocity Statistics for {phase}:\n")
            f.write(f"Mean magnitude: {np.mean(vel_magnitude):.6f}\n")
            f.write(f"Max magnitude: {np.max(vel_magnitude):.6f}\n")
            f.write(f"Min magnitude: {np.min(vel_magnitude):.6f}\n")
            f.write(f"Std magnitude: {np.std(vel_magnitude):.6f}\n")
            f.write(f"Number of vectors: {np.sum(mask)}/{vel.shape[0] * vel.shape[1]}\n")


# 使用示例
if __name__ == '__main__':
    # 创建处理器
    processor = PIVProcessor(tif_size=32)  # 改为30by30

    try:
        # 先测试单个Phase
        print("测试处理Phase 1...")
        info = processor.process_single_phase(1)
        print("\n处理完成！")
    except Exception as e:
        print(f"处理出错: {str(e)}")