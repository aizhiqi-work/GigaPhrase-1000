import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torchaudio
import numpy as np
import re
import sys
sys.path.append("/nvme01/openkws/qbyt")

from dataset.features import FeatureExtractor
from models.text.char_tokenizer import CharTokenizer
from g2p_en import G2p

test_dir="/nvme01/openkws/libriphrase/eval/"
# ---------------- Dataset ----------------
class LibriPhrasetTEST(Dataset):
    def __init__(
        self,
        test_dir=test_dir,
        csv=[
            "evaluation_set/libriphrase_diffspk_all_1word.csv",
            "evaluation_set/libriphrase_diffspk_all_2word.csv",
            "evaluation_set/libriphrase_diffspk_all_3word.csv",
            "evaluation_set/libriphrase_diffspk_all_4word.csv"
        ],
        types='easy',
        save_path='/nvme01/openkws/libriphrase/eval/evaluation_set/test_all_phrase.csv',
        augment=False,
    ):
        if not os.path.exists(save_path):
            self.data = pd.DataFrame(columns=['anchor_text', 'anchor', 'anchor_dur', 
                                              'comparison_text', 'comparison', 'comparison_dur', 
                                              'target', 'type'])
            for path in csv:
                n_word = os.path.join(test_dir, path)
                df = pd.read_csv(n_word)
                anc = df[['anchor_text', 'anchor', 'anchor_dur', 'comparison_text', 'comparison', 
                          'comparison_dur', 'target', 'type']]
                com = df[['comparison_text', 'comparison', 'comparison_dur', 'anchor_text', 'anchor', 
                          'anchor_dur', 'target', 'type']]
                self.data = self.data._append(
                    anc.rename(columns={y: x for x, y in zip(self.data.columns, anc.columns)}),
                    ignore_index=True
                )
                self.data = self.data._append(
                    com.rename(columns={y: x for x, y in zip(self.data.columns, com.columns)}),
                    ignore_index=True
                )
            self.data.to_csv(save_path, index=False)
        else:
            self.data = pd.read_csv(save_path)

        if types == 'easy':
            self.data = self.data.loc[self.data['type'].isin(['diffspk_easyneg', 'diffspk_positive'])]
        elif types == 'hard':
            self.data = self.data.loc[self.data['type'].isin(['diffspk_hardneg', 'diffspk_positive'])]
        
        self.data = self.data.values.tolist()
        self.test_dir = test_dir

        self.tokenizer = CharTokenizer('/nvme01/openkws/wenet/examples/librispeech-g2p/s0/data/dict/lang_char.txt', None, split_with_space=' ')
        self.feature_extractor = FeatureExtractor(augment=augment, wav_dir=test_dir)
        self.g2p = G2p()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _anchor_text, _anchor, _, _comparison_text, _comparison, _, _target, _ = self.data[index]
        
        anchor = _anchor_text
        anchor_phones = self.g2p(re.sub(r'[^\w\s]', '', _anchor_text.lower()))
        anchor_g2p = ' '.join([phone for phone in anchor_phones if phone != ' '])

        query_wav = _comparison
        feats = self.feature_extractor.process(query_wav)  # {'feat': [T, 80]}

        _, anchor_seq = self.tokenizer.tokenize(anchor_g2p)

        sample = {
            "anchor_seq": torch.tensor(anchor_seq, dtype=torch.long),  # text
            "feat": feats['feat'],  # audio tensor [T, 80]
            "label": torch.tensor(_target, dtype=torch.long),
            "wav_path": query_wav  # 原始音频路径
        }
        return sample


# ---------------- Collate_fn ----------------
def save_collate_fn(batch):
    """
    保存每条样本的 feat 为 .npy，文件路径与原音频一致，只加 .npy 后缀
    """
    for item in batch:
        feat = item['feat'].numpy()
        wav_path = item['wav_path']
        save_path = os.path.join(test_dir, wav_path).replace('.wav', '.npy')
        np.save(save_path, feat.astype(np.float32))
    
    return torch.tensor([item['label'] for item in batch])


# ---------------- Main ----------------
if __name__ == '__main__':
    test_dataset = LibriPhrasetTEST(types='easy')
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=256,
        num_workers=8,
        shuffle=False,
        collate_fn=save_collate_fn
    )

    from tqdm import tqdm
    for _ in tqdm(dataloader, desc="Saving fbank features"):
        pass

    print("全部 feat 已保存完毕！")
