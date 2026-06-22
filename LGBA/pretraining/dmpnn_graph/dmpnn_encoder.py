import torch
import torch.nn as nn
import torch.nn.functional as F

class DMPNNEncoder(nn.Module):
    """
    Directed Message Passing Neural Network (D-MPNN) encoder (vectorized).
    输入：
      - atom_features: Tensor[N, Fa]
      - bond_features: Tensor[E, Fb]（含双向边）
      - edge_index: LongTensor[2, E]
    输出：
      - mol_embed: Tensor[hidden_dim]
    """
    def __init__(self, atom_feat_dim, bond_feat_dim, hidden_dim, depth):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth

        # edge 初始化和更新
        self.edge_init   = nn.Linear(atom_feat_dim + bond_feat_dim, hidden_dim)
        self.edge_update = nn.Linear(2 * hidden_dim, hidden_dim)
        # 节点读出
        self.node_readout = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, atom_features, bond_features, edge_index):
        # 确保都在同一设备
        device = atom_features.device
        src, dst = edge_index  # 都是 [E]

        # 1) 初始化每条边的隐藏 h [E, H]
        init_input = torch.cat([atom_features[src], bond_features], dim=-1)
        h = F.relu(self.edge_init(init_input))

        # 2) 消息传递迭代
        E = h.size(0)
        # 假设反向边索引关系是 e_rev = e ^ 1（图构建时需保证双向边是成对相邻加入）
        rev_idx = torch.arange(E, device=device, dtype=torch.long) ^ 1

        for _ in range(self.depth):
            # 2.1) 计算每个节点的 “所有入边” 隐藏和 [N, H]
            N = atom_features.size(0)
            sum_in = torch.zeros(N, self.hidden_dim, device=device)
            sum_in = sum_in.index_add(0, dst, h)

            # 2.2) 对每条边，取 src 节点的入边和 `sum_in[src]`，减去它自己的反向边 h[e_rev]
            m = sum_in[src] - h[rev_idx]

            # 2.3) 批量更新所有边
            h = F.relu(self.edge_update(torch.cat([h, m], dim=-1)))

        # 3) 节点读出：将所有指向某节点的边聚合到节点上
        atom_embed = torch.zeros_like(sum_in)  # [N, H]
        atom_embed = atom_embed.index_add(0, dst, h)
        atom_embed = F.relu(self.node_readout(atom_embed))

        # 4) 全局读出（简单求和）
        mol_embed = atom_embed.sum(dim=0)
        return mol_embed

# 测试
if __name__ == "__main__":
    from smiles_graph_preprocess import smiles_to_graph
    graph = smiles_to_graph("CCO")
    atom_f, bond_f, edge_idx = graph.atom_features, graph.bond_features, graph.edge_index
    encoder = DMPNNEncoder(atom_feat_dim=atom_f.size(1),
                           bond_feat_dim=bond_f.size(1),
                           hidden_dim=128,
                           depth=3).to(atom_f.device)
    mol_vec = encoder(atom_f, bond_f, edge_idx)
    print("Molecular embedding shape:", mol_vec.shape)
