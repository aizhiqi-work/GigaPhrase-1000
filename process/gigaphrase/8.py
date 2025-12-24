import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

# 读取原始 Parquet 文件
parquet_file = "/nvme01/openkws/libriphrase/counts/ls-gs-1460/aggregated_segments_with_g2p_distance.parquet"
df = pd.read_parquet(
    parquet_file, 
    columns=["ngram", "ngram_g2p", "clips", "distances"]
)

# 创建一个用于存储 clips 和 distances 的文件夹
output_dir = "/nvme01/openkws/libriphrase/counts/ls-gs-1460/processed_files"
os.makedirs(output_dir, exist_ok=True)

def save_ngram_data_to_files(ngram, clips_data, distances_data, output_dir):
    """
    将 clips 和 distances 数据保存为 .npy 文件，并返回文件路径
    """
    # 生成文件名
    clips_data = json.loads(clips_data)

    clips_filename = os.path.join(output_dir, f"{ngram}-{len(clips_data)}-clips.npy")
    distances_filename = os.path.join(output_dir, f"{ngram}-{len(distances_data)}-distances.npy")
    
    np.save(clips_filename, clips_data)
    np.save(distances_filename, distances_data)
    
    return clips_filename, distances_filename

# 准备更新后的数据
file_paths = []

# 遍历每一行，将 clips 和 distances 存储为小文件
for idx, row in tqdm(df.iterrows(), total=len(df)):
    ngram = row["ngram"]
    clips_data = row["clips"]
    distances_data = row["distances"]
    
    # 保存数据到文件
    clips_file, distances_file = save_ngram_data_to_files(ngram, clips_data, distances_data, output_dir)
    
    # 将文件路径添加到列表
    file_paths.append((clips_file, distances_file))

# 更新 DataFrame，将 clips 和 distances 替换为文件路径
df["clips_file"] = [file_paths[i][0] for i in range(len(file_paths))]
df["distances_file"] = [file_paths[i][1] for i in range(len(file_paths))]

# 删除原来的 clips 和 distances 列
df = df.drop(columns=["clips", "distances"])

# 保存更新后的 DataFrame 到新的 Parquet 文件
output_parquet_file = "/nvme01/openkws/libriphrase/counts/ls-gs-1460/processed_data.parquet"
df.to_parquet(output_parquet_file, index=False)

print(f"处理后的数据已经保存到 {output_parquet_file}")
