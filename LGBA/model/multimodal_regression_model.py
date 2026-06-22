# multimodal_regression_model.py

import torch
import torch.nn as nn
from lgba.model.awd_lstm_encoder_dual_vocab import AWDLSTMEncoder
from lgba.model.dmpnn_wrapper_encoder import DMPNNWrapperEncoder
from lgba.model.cross_modal_fusion import CrossModalAttentionFusion, UniAttnAWD2DMPNN, UniAttnDMPNN2AWD


class MultiModalRegressionModel(nn.Module):
    def __init__(self,
                 spe_vocab_path,
                 awd_vocab_path,
                 awd_model_path,
                 dmpnn_model_path,
                 fusion_dim=256,
                 mlp_hidden_dim=128,
                 dropout=0.1,
                 device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- 子模块 ---
        self.awd_encoder = AWDLSTMEncoder(
            spe_vocab_path=spe_vocab_path,
            awd_vocab_path=awd_vocab_path,
            model_path=awd_model_path
        )

        self.dmpnn_encoder = DMPNNWrapperEncoder(
            model_path=dmpnn_model_path,
            hidden_dim=128,
            depth=3,
            device=self.device
        )

        self.fusion = CrossModalAttentionFusion(
            awd_dim=400,
            dmpnn_dim=128,
            hidden_dim=fusion_dim,
            dropout=dropout
        )

        # self.fusion = UniAttnAWD2DMPNN(
        #     awd_dim=400,
        #     dmpnn_dim=128,
        #     hidden_dim=fusion_dim,
        #     dropout=dropout
        # )

        # self.fusion = UniAttnDMPNN2AWD(
        #     awd_dim=400,
        #     dmpnn_dim=128,
        #     hidden_dim=fusion_dim,
        #     dropout=dropout
        # )

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1)  # 输出标量
        )

    def forward(self, smiles_list):
        """
        输入：list[str]，一批 SMILES
        输出：Tensor[B, 1]，预测值
        """
        awd_vec = self.awd_encoder(smiles_list).to(self.device)       # [B, 400]
        dmpnn_vec = self.dmpnn_encoder(smiles_list).to(self.device)   # [B, 128]
        fused = self.fusion(awd_vec, dmpnn_vec)                        # [B, 256]
        output = self.mlp(fused)                                       # [B, 1]
        return output
