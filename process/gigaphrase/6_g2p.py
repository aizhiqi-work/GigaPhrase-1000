from datasets import Dataset
import pandas as pd
import json
from g2p_en import G2p  # 你可以根据需要选择合适的 G2P 库

# 加载 Parquet 文件
parquet_file = "/nvme01/openkws/libriphrase/counts/ls-gs-1460/aggregated_segments_by_ngram.parquet"
dataset = Dataset.from_parquet(parquet_file)

# 查看 dataset 结构
print(dataset)
# 初始化 G2P 模型
g2p = G2p()

# # 为每个 ngram 生成 G2P
# def generate_g2p_for_ngram(ngram):
#     return g2p(ngram)  # G2P 转换，将字符转换为音标

# 在 dataset 中为每个 ngram 生成 G2P 音标
def process_ngram_with_g2p(item):
    ngram = item["ngram"]
    phones = g2p(ngram)
    item["ngram_g2p"] = ' '.join([phone for phone in phones if phone != ' '])
    return item

# 使用 map 函数处理 dataset，为每个 ngram 添加 G2P 转录
mapped_dataset = dataset.map(process_ngram_with_g2p, num_proc=1, batched=False, load_from_cache_file=False, desc="g2p")

# 显示处理后的 dataset
print(mapped_dataset)

# 保存添加了 G2P 音标的新 Parquet 文件
out_parquet = "/nvme01/openkws/libriphrase/counts/ls-gs-1460/aggregated_segments_with_g2p.parquet"
mapped_dataset.to_parquet(out_parquet)

print(f"✅ G2P 转录结果已保存到 {out_parquet}")
