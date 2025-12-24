import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

import torchaudio
import numpy as np
import re
import sys
import torchaudio
from datasets import load_dataset
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
import json

class GIGASPEECH(Dataset):
    def __init__(self):
        file_path = "/nvme01/openkws/gigaspeech-raw/giga-train-1000.txt"
        self.dataset = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        key = self.dataset[index]['key']
        wav = self.dataset[index]['wav']
        txt = self.dataset[index]['text']

        waveform, sr = torchaudio.load(wav)

        save_path = wav.replace(
            "/nvme01/openkws/gigaspeech-raw/hf_cache", "/nvme01/openkws/gigaspeech-raw/mfa"
        )
        save_path = os.path.splitext(save_path)[0] + ".txt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for tag in ['<COMMA>', '<PERIOD>', '<QUESTIONMARK>', '<EXCLAMATIONPOINT>']:
            txt = txt.replace(tag, '')
        txt = txt.strip().lower()
        clean_txt = re.sub(r'[^a-z\s]', '', txt)

        return {
            'key': key,
            'wav': waveform,
            'txt': clean_txt,
            'clean_txt': clean_txt,
            'save_path': save_path
        }





import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    batch: List[dict], 每个元素包含:
      {
        'key': str,
        'wav': Tensor [1, T],  # 单声道音频
        'txt': str,
        'clean_txt': str
      }
    """
    keys = [b["key"] for b in batch]
    txts = [b["txt"] for b in batch]
    clean_txts = [b["clean_txt"] for b in batch]
    save_paths = [b["save_path"] for b in batch]

    # 取掉 channel 维度 [1, T] -> [T]
    wavs = [b["wav"].squeeze(0) for b in batch]

    # 记录真实长度
    wav_lens = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)

    # padding 到 [B, max_T]
    wavs_padded = pad_sequence(wavs, batch_first=True)  # [B, max_T]

    return {
        "key": keys,
        "wav": wavs_padded,   # [B, max_T]
        "wav_lens": wav_lens, # [B]
        "txt": txts,
        "clean_txt": clean_txts,
        "save_paths": save_paths
    }




import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np


def align(emission, tokens):
    """执行强制对齐"""
    targets = torch.tensor([tokens], dtype=torch.int32, device=emission.device)
    alignments, scores = torchaudio.functional.forced_align(emission, targets, blank=0)
    alignments, scores = alignments[0], scores[0]
    scores = scores.exp()
    return alignments, scores


def unflatten(list_, lengths):
    """将token级别的对齐结果重组为词级别"""
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i: i + l])
        i += l
    return ret


def _score(spans):
    """计算加权平均置信度分数"""
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


class Wrapper(LightningModule):
    def __init__(self):
        super().__init__()
        # 如果你需要处理音频，可以保留MMS模型
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model(with_star=False)
        self.DICTIONARY = bundle.get_dict(star=None)
        
        # 添加数据处理相关的属性
        self.processed_count = 0
        self.output_dir = "/nvme01/openkws/process/gigaphrase/processed_data"

    def test_step(self, batch, batch_idx):
        # 使用DDP处理数据的主要方法
        keys = batch['key']
        wavs = batch['wav']
        txts = batch['txt']
        clean_txts = batch['clean_txt']
        wav_lens = batch['wav_lens']
        save_paths = batch['save_paths']
            
        # 示例：使用MMS模型处理音频（如果需要）
        with torch.inference_mode():
            emissions, _ = self.model(wavs)

        batch_size = wavs.shape[0]
        ratio = wavs.size(1) / emissions.size(1)
        for i in range(batch_size):
            save_path = save_paths[i]
            emission = emissions[i].unsqueeze(0)
            txt = txts[i]
            clean_txt = clean_txts[i]

            transcript = clean_txt.split()
            ori_transcript = txt.split()

            # 转 token
            tokenized_transcript = []
            for word in transcript:
                for c in word:
                    tokenized_transcript.append(self.DICTIONARY[c])
            
            aligned_tokens, alignment_scores = align(emission, tokenized_transcript)
            token_spans = torchaudio.functional.merge_tokens(aligned_tokens, alignment_scores)
            word_spans = unflatten(token_spans, [len(word) for word in transcript])

            results = []
            for word, spans in zip(ori_transcript, word_spans):
                x0 = int(ratio * spans[0].start)
                x1 = int(ratio * spans[-1].end)
                results.append({
                    'word': word,
                    'score': _score(spans),
                    'start_time': round(x0 / 16000, 4),
                    'end_time': round(x1 / 16000, 4)
                })

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)



    def on_test_epoch_end(self):
        print(f"进程 {self.global_rank} 总共处理了 {self.processed_count} 个样本")

    def configure_optimizers(self):
        return None



if __name__ == "__main__":
    pl.seed_everything(2025)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    
    # 创建数据集和数据加载器
    test_dataset = GIGASPEECH()
    dataloader = DataLoader(
        test_dataset, 
        batch_size=32, 
        num_workers=16, 
        shuffle=False, 
        collate_fn=collate_fn
    ) 
    
    wrapper = Wrapper()
    
    # 配置Trainer用于数据处理
    trainer = Trainer(
        max_epochs=1,           # 只处理一次
        devices=4,              # 使用4个GPU
        accelerator='gpu',      # 使用GPU
        strategy='ddp',         # 使用分布式数据并行
        logger=False,           # 不需要日志
        enable_checkpointing=False,  # 不需要检查点
        enable_progress_bar=True,    # 显示进度条
        enable_model_summary=False,  # 不需要模型摘要
    )
    
    # 使用test而不是fit，因为我们只是处理数据而不是训练
    print("开始使用DDP并行处理数据...")
    trainer.test(wrapper, dataloaders=dataloader)
    print("数据处理完成！")