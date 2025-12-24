import os
import json
import warnings
from datasets import load_dataset
from tqdm import tqdm

warnings.filterwarnings('ignore')

file_path = "/nvme01/openkws/gigaspeech-raw/giga-train-1000.txt"
dataset = load_dataset('json', data_files=file_path, split='train')

def preprocess(item):
    key = item['key']
    wav = item['wav']
    txt = item['text']

    # 清理文本
    for tag in ['<COMMA>', '<PERIOD>', '<QUESTIONMARK>', '<EXCLAMATIONPOINT>']:
        txt = txt.replace(tag, '')
    txt = txt.strip().lower()

    # mfa 对齐文件路径
    mfa_path = wav.replace(
        "/nvme01/openkws/gigaspeech-raw/hf_cache",
        "/nvme01/openkws/gigaspeech-raw/mfa"
    )
    mfa_path = os.path.splitext(mfa_path)[0] + ".txt"

    if not os.path.exists(mfa_path):
        return None

    with open(mfa_path, "r", encoding="utf-8") as f:
        mfa = json.load(f)

    ori_txt = txt.split()

    mfa_txt = [item['word'] for item in mfa]

    # ['as', 'theyre', 'leaving', 'can', 'kash', 'pull', 'zahra', 'aside', 'really', 'quickly']
    # ['as', "they're", 'leaving', 'can', 'kash', 'pull', 'zahra', 'aside', 'really', 'quickly']

    if len(ori_txt) != len(mfa_txt):
        return None

    ori_mfa = []
    for i in range(len(ori_txt)):
        ori_mfa.append({
            'word': ori_txt[i],
            'start_time': mfa[i]['start_time'],
            'end_time': mfa[i]['end_time'],
            'score': mfa[i]['score'],
        })

    return {
        'id': key,
        'audio_path': wav,
        'transcript': txt,
        'words': ori_mfa,
    }

# 并行处理
processed = dataset.map(
    preprocess,
    num_proc=16,
    desc="Processing",
)

# 去掉 None（因为有些 mfa 不存在）
processed = processed.filter(lambda x: x is not None)

# 保存为 parquet
save_path = "/nvme01/openkws/libriphrase/counts/gs-1000/list.parquet"
processed.to_parquet(save_path)

print(f"✅ 保存完成: {save_path}")
