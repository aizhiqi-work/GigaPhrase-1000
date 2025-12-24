import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

# 读取原始 Parquet 文件
parquet_file = "/nvme01/openkws/libriphrase/counts/ls-gs-1460/processed_data.parquet"
df = pd.read_parquet(
    parquet_file, 
    columns=["ngram", "ngram_g2p", "clips_file", "distances_file"]
)


for idx, row in tqdm(df.iterrows(), total=len(df)):
    ngram = row["ngram"]
    clips_file = row["clips_file"]
    distances_file = row["distances_file"]

    clips_data = np.load(clips_file, allow_pickle=True)
    distances_data = np.load(distances_file, allow_pickle=True)

    print(ngram, clips_data[0], distances_data[0])

    if idx == 1000:
        break
