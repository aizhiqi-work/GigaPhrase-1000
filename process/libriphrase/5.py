from datasets import Dataset
import pandas as pd
from collections import defaultdict
import json
from tqdm import tqdm


# 原始 parquet 文件路径
parquet_file = "/nvme01/openkws/libriphrase/counts/ls-460/segments_by_ngram_list.parquet"
dataset = Dataset.from_parquet(parquet_file)

# 查看 dataset 结构
print(dataset)

# 创建一个字典来按 ngram 聚合 clips
ngram_dict = defaultdict(list)

# 遍历 dataset 中的每一行，按 ngram 聚集 clips
for item in tqdm(dataset):
    ngram = item["ngram"]
    # 直接创建 clips 字段，结合其他字段
    clip = {
        "audio_path": item["audio_path"],
        # "duration": item["duration"],
        # "end": item["end"],
        # "start": item["start"],
        # "sr": item["sr"],
        # "sentence_id": item["sentence_id"],
        # "subset": item["subset"],
        # "ori_audio_path": item["ori_audio_path"]
    }
    
    # 将 clips 添加到 ngram_dict 中
    ngram_dict[ngram].append(clip)

# 将结果转换为 DataFrame
all_ngram_data = []
for ngram, clips in ngram_dict.items():
    all_ngram_data.append({
        "ngram": ngram,
        "ngram_lens": len(ngram.split()),
        "clips": json.dumps(clips)  # 将 clips 转换为 JSON 格式存储
    })

# 转换为 pandas DataFrame 以方便查看
df_grouped = pd.DataFrame(all_ngram_data)

# random
# 20k
# df_grouped = df_grouped.sample(n=40000, random_state=2025)

# 查看聚合后的 DataFrame
print(df_grouped)

# 保存为 parquet 文件
out_parquet = "/nvme01/openkws/libriphrase/counts/ls-460/78k/aggregated_segments_by_ngram.parquet"
df_grouped.to_parquet(out_parquet, index=False)

print(f"✅ 聚合后的数据已保存到 {out_parquet}")
