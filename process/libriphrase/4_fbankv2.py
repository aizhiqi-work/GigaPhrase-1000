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

import os
from dataset.features import FeatureExtractor
from models.text.char_tokenizer import CharTokenizer
from g2p_en import G2p

class LibriPhraset(Dataset):
    def __init__(self):
        self.feature_extractor = FeatureExtractor(augment=False, wav_dir='')
        self.data = self.read_file_list('/nvme01/openkws_github/eval/data_wwd/neg')


    def read_file_list(self, folder_path, ext='.wav'):
        file_list = []
        for root, dirs, files in os.walk(folder_path):
            for _file in files:
                if _file.endswith(ext):
                    file_list.append(os.path.join(root, _file))
        return file_list


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):
        query_wav = self.data[index]
        save_path = query_wav.replace("neg", "neg-fbank").replace('.wav', '.npy')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            return torch.tensor(0)

        feat = self.feature_extractor.process(query_wav)['feat'].numpy()
        np.save(save_path, feat.astype(np.float32))
        return torch.tensor(0)


# ---------------- Main ----------------
if __name__ == '__main__':
    test_dataset = LibriPhraset()
    print(len(test_dataset))
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=256,
        num_workers=12,
        shuffle=False,
    )

    from tqdm import tqdm
    for _ in tqdm(dataloader, desc="Saving fbank features"):
        pass

    print("全部 feat 已保存完毕！")










#     def __getitem__(self, index):
#         _anchor_text, _anchor, _, _comparison_text, _comparison, _, _target, _ = self.data[index]
        
#         anchor = _anchor_text
#         anchor_phones = self.g2p(re.sub(r'[^\w\s]', '', _anchor_text.lower()))
#         anchor_g2p = ' '.join([phone for phone in anchor_phones if phone != ' '])

#         query_wav = _comparison
#         feats = self.feature_extractor.process(query_wav)  # {'feat': [T, 80]}

#         _, anchor_seq = self.tokenizer.tokenize(anchor_g2p)

#         sample = {
#             "anchor_seq": torch.tensor(anchor_seq, dtype=torch.long),  # text
#             "feat": feats['feat'],  # audio tensor [T, 80]
#             "label": torch.tensor(_target, dtype=torch.long),
#             "wav_path": query_wav  # 原始音频路径
#         }
#         return sample


# # ---------------- Collate_fn ----------------
# def save_collate_fn(batch):
#     """
#     保存每条样本的 feat 为 .npy，文件路径与原音频一致，只加 .npy 后缀
#     """
#     for item in batch:
#         feat = item['feat'].numpy()
#         wav_path = item['wav_path']
#         save_path = os.path.join(test_dir, wav_path).replace('.wav', '.npy')
#         np.save(save_path, feat.astype(np.float32))
    
#     return torch.tensor([item['label'] for item in batch])


# # ---------------- Main ----------------
# if __name__ == '__main__':
#     test_dataset = LibriPhrasetTEST(types='easy')
    
#     dataloader = DataLoader(
#         test_dataset,
#         batch_size=256,
#         num_workers=8,
#         shuffle=False,
#         collate_fn=save_collate_fn
#     )

#     from tqdm import tqdm
#     for _ in tqdm(dataloader, desc="Saving fbank features"):
#         pass

#     print("全部 feat 已保存完毕！")

