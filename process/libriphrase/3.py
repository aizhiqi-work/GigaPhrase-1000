import os
import json
import re
from itertools import chain
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

# --- 文本清洗函数 ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # 只保留 a-z 和空格
    return text.strip()

def is_valid_ngram(ngram):
    return not any("’" in word or "'" in word for word in ngram)


# --- 加载全局 n-gram 词表 ---
def load_ngram_dict(base_dir, min_count=10):
    ngram_sets = {1: set(), 2: set(), 3: set(), 4: set()}
    for n in range(1, 5):
        filename = os.path.join(base_dir, f"ngram_{n}.txt")
        if not os.path.exists(filename):
            continue
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                word, count = line.strip().split("\t")
                count = int(count)
                if count >= min_count:
                    ngram_sets[n].add(word)
    return ngram_sets


# --- 核心函数：逐条样本筛选 ---
def filter_example(example, ngram_sets):
    transcript = example.get('transcript', '')
    transcript = clean_text(transcript).split()
    timestamps = json.loads(example.get('words', '[]'))

    clips = []

    window_size = 4
    for start_idx in range(len(timestamps)):
        for window_len in range(1, window_size + 1):
            end_idx = start_idx + window_len - 1
            if end_idx < len(timestamps):
                # 获取时间窗口
                window_start = timestamps[start_idx]["start"]
                window_end = timestamps[end_idx]["end"]
                duration = window_end - window_start
                if 0.5 <= duration <= 2.0:
                    window_words = [timestamps[i]["word"] for i in range(start_idx, end_idx + 1)]
                    ngram_len = len(window_words)
                    ngram_str = ' '.join(window_words)
                    if is_valid_ngram(window_words) and ngram_str in ngram_sets[ngram_len]:
                        clips.append({
                            "ngram": ngram_str,
                            "start": window_start,
                            "end": window_end,
                            "duration": duration
                        })

    # 返回统一格式：一句话一条
    return {
        "id": example["id"],
        "subset": example["subset"],
        "audio_path": example["audio_path"],
        "transcript": example["transcript"],
        "clips": clips
    }



# --- 主程序 ---
if __name__ == "__main__":
    parquet_file = "/nvme01/openkws/libriphrase/counts/ls-460/list.parquet"
    dataset_df = pd.read_parquet(parquet_file)
    dataset = Dataset.from_pandas(dataset_df)

    # 加载全局高频 n-gram 集合
    ngram_sets = load_ngram_dict("/nvme01/openkws/libriphrase/counts/ls-460", min_count=10)
    
    result_dataset = dataset.map(
        lambda ex: filter_example(ex, ngram_sets),
        num_proc=16,
        batched=False,
        load_from_cache_file=False,
        desc="筛选符合条件的句子片段"
    )

    df = result_dataset.to_pandas()
    out_file = "/nvme01/openkws/libriphrase/counts/ls-460/matched_sentences.parquet"
    df.to_parquet(out_file, index=False)
    print(f"✅ 已保存 {len(df)} 条句子结果到 {out_file}")
