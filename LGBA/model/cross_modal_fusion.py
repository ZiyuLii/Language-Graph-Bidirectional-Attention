# cross_modal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionFusion(nn.Module):
    def __init__(self,
                 awd_dim=400,
                 dmpnn_dim=128,
                 hidden_dim=256,
                 dropout=0.1):
        super().__init__()
        self.awd_proj = nn.Linear(awd_dim, hidden_dim)
        self.dmpnn_proj = nn.Linear(dmpnn_dim, hidden_dim)

        # AWD → DMPNN 的注意力
        self.attn_awd_to_dmpnn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=4,
                                          batch_first=True,
                                          dropout=dropout)
        
        # DMPNN → AWD 的注意力
        self.attn_dmpnn_to_awd = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=4,
                                          batch_first=True,
                                          dropout=dropout)

        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 双向拼接
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, awd_repr, dmpnn_repr):
        """
        awd_repr:   Tensor[B, 400]
        dmpnn_repr: Tensor[B, 128]
        return:     Tensor[B, hidden_dim]
        """
        B = awd_repr.size(0)
        
        # AWD → DMPNN 注意力
        q_awd = self.awd_proj(awd_repr).unsqueeze(1)    # [B, 1, H]
        kv_dmpnn = self.dmpnn_proj(dmpnn_repr).unsqueeze(1)  # [B, 1, H]
        
        fused_awd_to_dmpnn, _ = self.attn_awd_to_dmpnn(q_awd, kv_dmpnn, kv_dmpnn)  # [B, 1, H]
        fused_awd_to_dmpnn = fused_awd_to_dmpnn.squeeze(1)  # [B, H]
        
        # DMPNN → AWD 注意力
        q_dmpnn = self.dmpnn_proj(dmpnn_repr).unsqueeze(1)  # [B, 1, H]
        kv_awd = self.awd_proj(awd_repr).unsqueeze(1)    # [B, 1, H]
        
        fused_dmpnn_to_awd, _ = self.attn_dmpnn_to_awd(q_dmpnn, kv_awd, kv_awd)  # [B, 1, H]
        fused_dmpnn_to_awd = fused_dmpnn_to_awd.squeeze(1)  # [B, H]
        
        # 双向融合
        combined = torch.cat([fused_awd_to_dmpnn, fused_dmpnn_to_awd], dim=1)  # [B, 2H]
        
        # 残差 + MLP
        out = self.mlp(combined)  # [B, H]
        return out
    

class UniAttnAWD2DMPNN(nn.Module):
    def __init__(self, awd_dim=400, dmpnn_dim=128, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.awd_proj = nn.Linear(awd_dim, hidden_dim)
        self.dmpnn_proj = nn.Linear(dmpnn_dim, hidden_dim)

        self.attn_awd_to_dmpnn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                       num_heads=4,
                                                       batch_first=True,
                                                       dropout=dropout)

        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, awd_feat, dmpnn_feat):
        # 维度变换
        awd_proj = self.awd_proj(awd_feat)      # [B, L1, H]
        dmpnn_proj = self.dmpnn_proj(dmpnn_feat)  # [B, L2, H]

        # 单向注意力：AWD 作为 query，DMPNN 作为 key/value
        attn_output, _ = self.attn_awd_to_dmpnn(query=awd_proj, key=dmpnn_proj, value=dmpnn_proj)

        # 残差 + 归一化 + MLP
        out = self.norm(attn_output + awd_proj)
        out = self.mlp(out)

        return out  # 可直接池化后用于预测

class UniAttnDMPNN2AWD(nn.Module):
    def __init__(self, awd_dim=400, dmpnn_dim=128, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.awd_proj = nn.Linear(awd_dim, hidden_dim)
        self.dmpnn_proj = nn.Linear(dmpnn_dim, hidden_dim)

        self.attn_dmpnn_to_awd = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                       num_heads=4,
                                                       batch_first=True,
                                                       dropout=dropout)

        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, awd_feat, dmpnn_feat):
        awd_proj = self.awd_proj(awd_feat)
        dmpnn_proj = self.dmpnn_proj(dmpnn_feat)

        # 单向注意力：DMPNN 作为 query，AWD 作为 key/value
        attn_output, _ = self.attn_dmpnn_to_awd(query=dmpnn_proj, key=awd_proj, value=awd_proj)

        out = self.norm(attn_output + dmpnn_proj)
        out = self.mlp(out)

        return out
