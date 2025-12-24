import os
import json
import torchaudio
from datasets import Dataset
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# 输出目录
OUT_DIR = "/nvme01/openkws/libriphrase/segments/GP-1000"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# 切分函数（每个 example 内累积 ngram clips）
# ----------------------------
def process_sentence(example):
    audio_path = example["audio_path"]
    sentence_id = example["id"]

    clips = example["clips"]
    if isinstance(clips, str):
        clips = json.loads(clips)

    # ngram -> list of clips
    ngrams = []

    try:
        waveform, sr = torchaudio.load(audio_path)
        for i, clip in enumerate(clips):
            start = clip["start"]
            end = clip["end"]
            start_frame = int(start * sr)
            end_frame = int(end * sr)
            segment = waveform[:, start_frame:end_frame]
            ngram = clip["ngram"].replace(" ", "_")

            # 存储音频
            seg_dir = os.path.join(OUT_DIR, ngram)
            os.makedirs(seg_dir, exist_ok=True)
            seg_filename = f"{sentence_id}_{i:03d}.wav"

            seg_path = os.path.join(seg_dir, seg_filename)
            torchaudio.save(seg_path, segment, sr)

            # 添加到 ngram_dict
            ngrams.append({
                "ngram": ngram,
                "audio_path": seg_path,
                "sentence_id": sentence_id,
                "start": start,
                "end": end,
                "sr": sr,
                "duration": round((end_frame - start_frame) / sr, 3)
            })

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")

    # map 函数必须返回 dict 或 None
    # 这里直接返回 ngram_dict 方便后续合并
    return {"ngrams": ngrams}


# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    parquet_file = "/nvme01/openkws/libriphrase/counts/gs-1000/matched_sentences.parquet"
    df = pd.read_parquet(parquet_file)
    dataset = Dataset.from_pandas(df)
    
    print(f"✅ 加载完成，共 {len(dataset)} 句")
    
    # 多进程 map
    mapped_dataset = dataset.map(
        process_sentence,
        num_proc=64,
        batched=False,
        load_from_cache_file=False,
        remove_columns=dataset.column_names,
        desc="切分音频"
    )

    print(f"✅ 切分完成，共 {len(mapped_dataset)} 条 segment")

    from collections import defaultdict
    from tqdm import tqdm

    all_ngram = []
    for item in tqdm(mapped_dataset, desc="聚合 ngram"):
        ngram_sub = item.get("ngrams", [])
        all_ngram.extend(ngram_sub)

    df_grouped = pd.DataFrame(all_ngram)
    df_grouped["ngram"] = df_grouped["ngram"].apply(lambda x: x.replace("_", " "))
    df_grouped["audio_path"] = df_grouped["audio_path"].apply(lambda x: x.replace(OUT_DIR, "GP-1000"))
    # 保存 parquet
    out_parquet = "/nvme01/openkws/libriphrase/counts/gs-1000/segments_by_ngram_list.parquet"
    df_grouped.to_parquet(out_parquet, index=False)
    print(f"✅ 按 ngram 聚合后的 metadata 已保存到 {out_parquet}")
