from datasets import load_from_disk, disable_caching
from collections import Counter
import re
import os
from tqdm import tqdm
from itertools import chain
from datasets import Dataset
import json



# 文本清洗：转小写 + 去除非字母字符
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # 只保留 a-z 和空格
    return text.strip()

# 用于清理包含 ’ 字符的 n-gram
def is_valid_ngram(ngram):
    return not any("’" in word or "'" in word for word in ngram)


# 单样本处理函数：返回每个 n-gram 的列表（字符串列表）
def process_example(example):
    transcript = example.get('transcript', example.get('text', ''))
    transcript = clean_text(transcript).split()
    timestamps = json.loads(example.get('words', '[]'))  # 防止 None 或空值导致出错

    # 滑动窗口，检查每个组合的时间长度
    valid_windows = []
    window_size = 4  # 最多取4个词的组合

    # 遍历每个可能的窗口组合
    for start_idx in range(len(timestamps)):
        for window_len in range(1, window_size + 1):  # 窗口长度为1到4
            end_idx = start_idx + window_len - 1
            if end_idx < len(timestamps):
                # 获取当前窗口的时间
                window_start = timestamps[start_idx]["start"]
                window_end = timestamps[end_idx]["end"]
                duration = window_end - window_start

                # 获取窗口内的词
                window_words = [timestamps[i]["word"] for i in range(start_idx, end_idx + 1)]
                concat_len = len(''.join(window_words))  # 拼接长度

                # 最小长度要求：1个词>=2，2个词>=4，3个词>=6 ...
                min_len = 2 * window_len

                # 时间和长度过滤
                if 0.5 <= duration <= 2.0 and concat_len >= min_len:
                    valid_windows.append(window_words)

    # 分类保存 n-grams
    ngrams = {'ngram_1': [], 'ngram_2': [], 'ngram_3': [], 'ngram_4': []}
    for window in valid_windows:
        ngram_len = len(window)
        if 1 <= ngram_len <= 4:
            ngram_str = ' '.join(window)
            if is_valid_ngram(window):  # 过滤掉包含 ’ 字符的 n-gram
                ngrams[f'ngram_{ngram_len}'].append(ngram_str)
    
    return ngrams

if __name__ == '__main__':
    print("📂 加载数据集...")
    import pandas as pd

    parquet_file = "/nvme01/openkws/libriphrase/counts/ls-460/list.parquet"
    dataset_df = pd.read_parquet(parquet_file)

    print(f"✅ 数据集加载完成，共 {len(dataset_df)} 个样本")
    
    # 将 pandas DataFrame 转换为 Hugging Face Dataset 格式
    dataset = Dataset.from_pandas(dataset_df)

    print(f"✅ 数据集加载完成，共 {len(dataset)} 个样本")
    print("🚀 开始逐条处理，多进程提取 n-gram...")

    # 使用 map 多进程处理，逐条（非 batched）
    result_dataset = dataset.map(
        process_example,
        num_proc=8,                    # 多进程，可调整为你的 CPU 核心数
        batched=False,                 # 一条一条处理
        load_from_cache_file=False,    # 禁用缓存
        desc="提取 n-gram (逐条)",       # 进度条提示
    )

    print("📊 正在统计总频率...")

    # 使用 chain.from_iterable 高效展平 list of lists
    final_counters = {
        1: Counter(chain.from_iterable(result_dataset['ngram_1'])),
        2: Counter(chain.from_iterable(result_dataset['ngram_2'])),
        3: Counter(chain.from_iterable(result_dataset['ngram_3'])),
        4: Counter(chain.from_iterable(result_dataset['ngram_4'])),
    }

    # 创建输出目录
    os.makedirs("ngram_results", exist_ok=True)

    # 保存结果到文件，只保留频率 >= 2 的项
    print("\n💾 正在保存结果...")
    for n in range(1, 5):
        filename = f"/nvme01/openkws/libriphrase/counts/ls-460/ngram_{n}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            for word, count in final_counters[n].most_common():
                if count >= 10:
                    f.write(f"{word}\t{count}\n")
        print(f"✅ 已保存 {n}-gram 到 {filename}")
