import torch
from typing import Tuple
from lgba.pretraining.dmpnn_graph.smiles_graph_preprocess import MolecularGraph

def drop_edges(graph: MolecularGraph, drop_rate: float = 0.1) -> MolecularGraph:
    """
    随机删除无向边对（CPU 上运行）。
    """
    edge_idx = graph.edge_index      # CPU 上的 LongTensor [2, E]
    E = edge_idx.size(1)
    num_pairs = E // 2

    # CPU 上生成随机 mask
    pair_ids = torch.arange(E) // 2           # [E], CPU
    mask_pairs = torch.rand(num_pairs) >= drop_rate  # [num_pairs] CPU bool
    keep_mask = mask_pairs[pair_ids]          # [E] CPU bool

    new_edge_idx   = edge_idx[:, keep_mask]
    new_bond_feats = graph.bond_features[keep_mask]

    return MolecularGraph(graph.atom_features, new_bond_feats, new_edge_idx)


def mask_nodes(graph: MolecularGraph, mask_rate: float = 0.15) -> Tuple[MolecularGraph, torch.Tensor]:
    """
    随机遮蔽节点特征（CPU 上运行）。
    """
    atom_feats = graph.atom_features       # [N, F], CPU Tensor
    N = atom_feats.size(0)

    # 在 CPU 上生成 mask
    rand_vals = torch.rand(N)
    mask_idx  = (rand_vals < mask_rate).nonzero(as_tuple=False).squeeze(1)  # CPU LongTensor

    new_atom = atom_feats.clone()
    new_atom[mask_idx] = 0.0

    return MolecularGraph(new_atom, graph.bond_features, graph.edge_index), mask_idx


def augment_graph(graph: MolecularGraph,
                  drop_rate: float = 0.1,
                  mask_rate: float = 0.15) -> Tuple[MolecularGraph, torch.Tensor]:
    """
    CPU 上增强：先删除边再遮蔽节点。
    """
    gd = drop_edges(graph, drop_rate)
    gm, mask_idx = mask_nodes(gd, mask_rate)
    return gm, mask_idx


def augment_graph_pair(graph: MolecularGraph,
                       drop_rate: float = 0.1,
                       mask_rate: float = 0.15
                      ) -> Tuple[Tuple[MolecularGraph, torch.Tensor],
                                 Tuple[MolecularGraph, torch.Tensor]]:
    """
    CPU 上做两次增强，可复现。
    """
    cpu_state = torch.get_rng_state()

    torch.manual_seed(42)
    g1, m1 = augment_graph(graph, drop_rate, mask_rate)

    torch.manual_seed(24)
    g2, m2 = augment_graph(graph, drop_rate, mask_rate)

    torch.set_rng_state(cpu_state)
    return (g1, m1), (g2, m2)
