# awd_lstm_encoder_dual_vocab.py

import torch
import torch.nn as nn
from fastai.text.models.awdlstm import AWD_LSTM
import pickle


class AWDLSTMEncoder(nn.Module):
    def __init__(self,
                 spe_vocab_path,   # SPE 分词词表（list[str]）
                 awd_vocab_path,   # AWD-LSTM 训练用词表（list[str]）
                 model_path,       # AWD-LSTM 编码器权重 .pth
                 emb_sz=400,
                 n_hid=1152,
                 n_layers=3,
                 pad_token=1,
                 drop_mult=0.5,
                 max_len=72):
        super().__init__()
        self.pad_token = pad_token
        self.max_len = max_len

        # ===== 加载 SPE 词表 =====
        with open(spe_vocab_path, "rb") as f:
            spe_list = pickle.load(f)
        self.spe_tokens = sorted(spe_list, key=len, reverse=True)  # 用于贪心分词

        # ===== 加载 AWD 词表 =====
        with open(awd_vocab_path, "rb") as f:
            awd_list = pickle.load(f)
        self.token_to_index = {tok: i for i, tok in enumerate(awd_list)}  # 用于 token → ID
        self.vocab_size = len(self.token_to_index)

        # ===== 构建 AWD-LSTM 编码器 =====
        self.encoder = AWD_LSTM(
            vocab_sz=self.vocab_size,
            emb_sz=emb_sz,
            n_hid=n_hid,
            n_layers=n_layers,
            pad_token=pad_token,
            bidir=False
        )

        # ===== 加载预训练参数 =====
        state_dict = torch.load(model_path, map_location="cpu")
        # 过滤掉大小不匹配的参数
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in self.encoder.state_dict() and v.shape == self.encoder.state_dict()[k].shape:
                filtered_state_dict[k] = v
        self.encoder.load_state_dict(filtered_state_dict, strict=False)

    def spe_tokenize(self, smiles: str):
        """
        使用 SPE 词表进行贪心分词（返回 token list）
        """
        tokens = []
        i, n = 0, len(smiles)
        while i < n:
            matched = False
            for tok in self.spe_tokens:
                L = len(tok)
                if i + L <= n and smiles[i:i+L] == tok:
                    tokens.append(tok)
                    i += L
                    matched = True
                    break
            if not matched:
                tokens.append(smiles[i])
                i += 1
        return tokens

    def encode_tokens(self, tokens: list[str]):
        """
        将 token list 转换为 token ID list
        """
        unk = self.token_to_index.get("<UNK>", -1)
        encoded = [self.token_to_index.get(tok, unk) for tok in tokens]
        # 过滤掉UNK标记
        return [idx for idx in encoded if idx != -1]

    def batch_tokenize_and_encode(self, smiles_list):
        """
        SMILES 列表 → padded LongTensor [B, T]
        """
        token_id_lists = []
        for smi in smiles_list:
            toks = self.spe_tokenize(smi)
            ids = self.encode_tokens(toks)
            if len(ids) < self.max_len:
                ids += [self.pad_token] * (self.max_len - len(ids))
            else:
                ids = ids[:self.max_len]
            token_id_lists.append(ids)
        return torch.tensor(token_id_lists, dtype=torch.long)

    def forward(self, smiles_list):
        """
        输入：str 或 list[str]（SMILES）
        输出：Tensor[B, H]，默认输出最后一个时间步的句向量
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        x = self.batch_tokenize_and_encode(smiles_list)  # [B, T]
        raw_outputs = self.encoder(x)                 # [B, T, H]
        return raw_outputs[:, -1, :]                 # [B, H]
