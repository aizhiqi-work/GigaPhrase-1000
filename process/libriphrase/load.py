import pandas as pd
import json
import time
from tqdm import tqdm

tqdm.pandas()  # 注册 pandas tqdm 扩展
parquet_file = "/nvme01/openkws/libriphrase/counts/ls-100/aggregated_segments_with_g2p_distance.parquet"

# 只读取需要的列
df = pd.read_parquet(
    parquet_file, 
    columns=["ngram", "ngram_g2p", "clips", "distances"]
)
# clips 本来是 str，解析为 list，并显示进度条
df["clips"] = df["clips"].progress_apply(json.loads)
df["distances"] = df["distances"].progress_apply(lambda x: x.tolist())

print(df)

# ngrams = df["ngram"].tolist()
# ngram2idx = {ngram: idx for idx, ngram in enumerate(ngrams)}

# target_ngram = "himself"

# if target_ngram in ngram2idx:
#     idx = ngram2idx[target_ngram]   # O(1)
#     row = df.iloc[idx]              # 直接取该行
#     print(f"\n信息 for ngram='{target_ngram}':")
#     print("clips:", row["clips"][:10])        # 打印前3个示例
#     print("distances:", row["distances"][:10])
# else:
#     print(f"{target_ngram} 不在 dataset 中")
