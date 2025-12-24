import os
import json
from pathlib import Path
from typing import List, Dict, Union
import torchaudio
import pandas as pd
from tqdm import tqdm
import tgt


# --- 全局常量 ---
INVALID_LABELS = {'', 'sp', 'sil', 'spn', 'ns', 'noise', 'laugh', 'breath', '<p>', '<unk>'}
AUDIO_EXTENSIONS = ('.flac', '.wav', '.mp3', '.aac')


# --- 工具函数 ---
def is_valid_label(text):
    """判断标签是否有效（不在 INVALID_LABELS 中）"""
    return text.strip().lower() not in INVALID_LABELS


def textgrid_to_json(textgrid_path, words_tier_name='words'):
    """
    读取 TextGrid 文件，提取 Words 层并过滤无效标签，返回 JSON 格式的数据。
    """
    try:
        tg = tgt.io.read_textgrid(textgrid_path, encoding='utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to read TextGrid file: {textgrid_path}") from e

    # 获取 Words 层
    words_tier = tg.get_tier_by_name(words_tier_name)
    if not words_tier:
        raise ValueError(f"Tier '{words_tier_name}' not found.")

    # 提取有效的 Words
    words = []
    for interval in words_tier.intervals:
        word = interval.text.strip()
        if is_valid_label(word):
            words.append({
                "start": round(interval.start_time, 3),
                "end": round(interval.end_time, 3),
                "word": word
            })

    return {"words": words}


def read_transcript_file_as_dict(file_path):
    """
    读取 .trans.txt 文件并转换为字典
    """
    transcript_dict = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                space_idx = line.find(' ')
                if space_idx > 0:
                    utt_id = line[:space_idx]
                    transcript = line[space_idx + 1:].strip()
                    transcript_dict[utt_id] = transcript
                    
    except Exception as e:
        print(f"读取文件出错: {e}")
        return {}
    
    return transcript_dict


def find_audio_textgrid_pairs_recursive_with_transcript(folder_path):
    """
    递归遍历文件夹，查找所有音频文件，并检查是否存在对应的 .TextGrid 文件和原始 transcript。
    返回包含音频路径、TextGrid 路径和转录文本的元组列表。
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    # 获取所有音频文件（递归）
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(folder.rglob(f"*{ext}"))
        audio_files.extend(folder.rglob(f"*{ext.upper()}"))

    # 快速查找 TextGrid 文件：stem -> Path
    textgrid_files = {file.stem: file for file in folder.rglob("*.TextGrid")}

    pairs = []

    for audio in tqdm(audio_files, desc="Processing Audio Files"):
        stem = audio.stem  # 如 '19_227_000001'

        # 查找 TextGrid 文件
        tg_file = textgrid_files.get(stem)
        textgrid_path = str(tg_file) if tg_file else None

        # 查找 transcript 文件
        parent_dir = audio.parent
        trans_filename = f"{'-'.join(stem.split('-')[:2])}.trans.txt"
        trans_file = parent_dir / trans_filename
        trans_dict = read_transcript_file_as_dict(trans_file)
        
        # 添加配对
        if stem in trans_dict:
            pairs.append((str(audio), textgrid_path, trans_dict[stem]))

    # 按音频路径排序
    pairs.sort(key=lambda x: x[0])
    return pairs


def process_all_data_to_parquet(input_folders: Union[str, List[str]], output_file: str):
    """
    主处理函数：将 LibriSpeech 数据集转换为一个 Parquet 文件
    """
    # 如果是单个文件夹，将其转换为列表形式
    if isinstance(input_folders, str):
        input_folders = [input_folders]

    rows = []
    
    # 处理每个文件夹
    for input_folder in input_folders:
        print(f"开始处理文件夹: {input_folder}")
        
        pairs = find_audio_textgrid_pairs_recursive_with_transcript(input_folder)
        total_pairs = len(pairs)
        print(f"总共找到 {total_pairs} 个音频文件")

        # 处理每个音频文件
        for audio_path, textgrid_path, transcript in tqdm(pairs, desc="Processing Audio Files"):
            if not textgrid_path or not transcript:
                continue  # 忽略缺少 TextGrid 或 transcript 的样本

            try:
                # 解析 TextGrid
                tg_data = textgrid_to_json(textgrid_path)

                # 构造行数据
                row = {
                    "subset": Path(audio_path).parts[-4],  # 如 train-clean-100
                    "id": Path(audio_path).stem,
                    "audio_path": audio_path,
                    "transcript": transcript.lower(),
                    "words": json.dumps(tg_data["words"]),
                }
                rows.append(row)

            except Exception as e:
                print(f"❌ Error processing {audio_path}: {e}")

    # 保存所有数据到 Parquet 文件
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(output_file, index=False)
        print(f"✅ Saved to {output_file}, rows: {len(df)}")
    else:
        print("⚠️ No valid data to save.")


# --- 主程序入口 ---
if __name__ == "__main__":
    folders = [
        "/server24/openkws/data/librispeech-raw/LibriSpeech/train-clean-100",
        "/server24/openkws/data/librispeech-raw/LibriSpeech/train-clean-360",  
        # 如果有多个文件夹，直接加到这个列表中
    ]
    output_parquet = "/nvme01/openkws/libriphrase/counts/ls-460/list.parquet"

    print(f"\n🚀 开始处理目录: {folders}")
    process_all_data_to_parquet(folders, output_parquet)
