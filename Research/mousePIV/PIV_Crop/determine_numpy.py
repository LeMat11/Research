#!/usr/bin/env python
import os
import numpy as np

def check_npy_shapes(npy_dir):
    shape_counts = {}
    total_count = 0

    for root, dirs, files in os.walk(npy_dir):
        for fname in files:
            if fname.endswith('.npy'):
                fpath = os.path.join(root, fname)
                total_count += 1

                try:
                    data = np.load(fpath)
                    shape = data.shape
                    shape_str = str(shape)
                    if shape_str not in shape_counts:
                        shape_counts[shape_str] = 0
                    shape_counts[shape_str] += 1
                except Exception as e:
                    print(f"读取 {fpath} 出错: {e}")

    print("\n=== NPY 形状统计 ===")
    print(f"总文件数: {total_count}")
    for shape_str, count in shape_counts.items():
        perc = (count/total_count)*100 if total_count>0 else 0
        print(f"形状 {shape_str} : {count} 文件 ({perc:.2f}%)")

if __name__ == "__main__":
    # 在 Windows 系统用 UNC 路径时，需要转义反斜杠或用原始字符串:
    npy_dir = r'\\10.24.62.51\Lab Shared\Project Mouse PIV\PreDataSet\32by32\dat_v2'
    check_npy_shapes(npy_dir)
