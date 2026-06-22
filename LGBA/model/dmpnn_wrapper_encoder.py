# dmpnn_wrapper_encoder.py

import torch
import torch.nn as nn
from lgba.pretraining.dmpnn_graph.smiles_graph_preprocess import smiles_to_graph
from lgba.pretraining.dmpnn_graph.dmpnn_encoder import DMPNNEncoder


class DMPNNWrapperEncoder(nn.Module):
    def __init__(self,
                 model_path,
                 atom_feat_dim=6,
                 bond_feat_dim=3,
                 hidden_dim=128,
                 depth=3,
                 device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = DMPNNEncoder(
            atom_feat_dim=atom_feat_dim,
            bond_feat_dim=bond_feat_dim,
            hidden_dim=hidden_dim,
            depth=depth
        ).to(self.device)

        # 加载训练好的权重
        self.encoder.load_state_dict(torch.load(model_path, map_location=self.device))
        self.encoder.eval()

    def forward(self, smiles_list):
        """
        输入：单个 SMILES 字符串或 list[str]
        输出：Tensor [B, hidden_dim]
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        batch_embeddings = []
        for smi in smiles_list:
            graph = smiles_to_graph(smi)
            a, b, e = graph.atom_features.to(self.device), graph.bond_features.to(self.device), graph.edge_index.to(self.device)
            h = self.encoder(a, b, e)
            batch_embeddings.append(h)

        return torch.stack(batch_embeddings, dim=0)  # [B, H]
