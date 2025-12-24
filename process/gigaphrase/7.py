import pandas as pd
import re
import Levenshtein
from tqdm import tqdm
from multiprocessing import Pool
from datasets import Dataset


# 加载数据集
parquet_file = "/nvme01/openkws/libriphrase/counts/ls-gs-1460/aggregated_segments_with_g2p.parquet"
df = pd.read_parquet(parquet_file)
dataset = Dataset.from_parquet(parquet_file)
ngram_list = df['ngram'].tolist()
ngram_lens_list = df['ngram_lens'].tolist()
ngram_g2p_list = df['ngram_g2p'].tolist()

# 计算 Levenshtein 距离的函数
def calculate_levenshtein_distances(seq1, seq2, digits=True):
    """
    计算两个音标序列的 Levenshtein 距离
    1. 保留数字的 Levenshtein 距离
    2. 去掉数字后的 Levenshtein 距离
    """
    if digits:
        seq1 = re.sub(r'\d+', '', seq1)
        seq2 = re.sub(r'\d+', '', seq2)
    
    # 计算 Levenshtein 距离
    distance = Levenshtein.distance(seq1, seq2)
    
    # 归一化相似度（Levenshtein 距离与最大长度的比值）
    similarity = 1 - (distance / max(len(seq1), len(seq2)))
    return similarity

def process_distance(example):
    target_ngram = example['ngram_g2p']
    target_ngram_len = example['ngram_lens']
    distances = []
    
    # 计算每个 ngram 与目标 ngram 的距离
    for i in range(len(ngram_list)):
        ngram = ngram_g2p_list[i]
        ngram_len = ngram_lens_list[i]
        if ngram_len != target_ngram_len:
            continue
        distances.append({
            'ngram': ngram_list[i],
            'distance': round(calculate_levenshtein_distances(target_ngram, ngram, digits=True), 3)
        })
    
    # 按距离排序，从大到小
    distances = sorted(distances, key=lambda x: x['distance'], reverse=True)
    distances = [d for d in distances if d['distance'] <= 0.95]
    example['distances'] = distances[:100]
    return example


# 多进程 map
mapped_dataset = dataset.map(
    process_distance,
    num_proc=128,  # 使用32个进程进行并行处理
    batched=False,  # 每个样本单独处理
    load_from_cache_file=False,  # 不从缓存加载
    desc="计算 Levenshtein 距离并筛选"
)

# 保存 parquet
out_parquet = "/nvme01/openkws/libriphrase/counts/ls-gs-1460/aggregated_segments_with_g2p_distance.parquet"
mapped_dataset.to_parquet(out_parquet)
print(f"✅ 按 ngram 聚合后的 metadata 已保存到 {out_parquet}")
