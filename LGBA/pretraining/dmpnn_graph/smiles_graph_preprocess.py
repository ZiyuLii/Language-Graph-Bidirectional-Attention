import torch
from torch.utils.data import Dataset
from rdkit import Chem
import csv

# 原子和键特征提取
ATOM_FEATURES = {
    'atomic_num':     lambda atom: atom.GetAtomicNum(),
    'degree':         lambda atom: atom.GetDegree(),
    'formal_charge':  lambda atom: atom.GetFormalCharge(),
    'chiral_tag':     lambda atom: int(atom.GetChiralTag()),
    'num_hs':         lambda atom: atom.GetTotalNumHs(),
    'is_aromatic':    lambda atom: int(atom.GetIsAromatic())
}

BOND_FEATURES = {
    'bond_type':      lambda bond: int(bond.GetBondTypeAsDouble()),
    'is_conjugated':  lambda bond: int(bond.GetIsConjugated()),
    'is_in_ring':     lambda bond: int(bond.IsInRing())
}

class MolecularGraph:
    """
    保存已经转为张量并 pin_memory 的图数据：
      - atom_features: Tensor[N, natom_feats], pinned
      - bond_features: Tensor[E, nbond_feats], pinned
      - edge_index:    LongTensor[2, E], pinned
    """
    def __init__(self, atom_feats, bond_feats, edge_index):
        # 假设已传入 pin_memory 的张量
        self.atom_features = atom_feats
        self.bond_features = bond_feats
        self.edge_index    = edge_index

def smiles_to_graph(smiles: str) -> MolecularGraph:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 1) 原子特征列表
    atom_list = []
    for atom in mol.GetAtoms():
        atom_list.append([fn(atom) for fn in ATOM_FEATURES.values()])
    atom_feats = torch.tensor(atom_list, dtype=torch.float32).pin_memory()

    # 2) 键特征 & 有向边
    bond_list = []
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = [fn(bond) for fn in BOND_FEATURES.values()]
        # 拆成双向
        bond_list.append(feats); edges.append((i, j))
        bond_list.append(feats); edges.append((j, i))

    bond_feats = torch.tensor(bond_list, dtype=torch.float32).pin_memory()
    # 转置成 [2, E]
    edge_index = torch.tensor(edges, dtype=torch.long).t().pin_memory()

    return MolecularGraph(atom_feats, bond_feats, edge_index)

class GraphDataset(Dataset):
    """
    用于 DataLoader，直接加载并缓存所有图，
    每个 item 返回已 pin 的 Tensor，可直接 to(device)。
    """
    def __init__(self, csv_path: str, smiles_col: str = 'SMILES'):
        self.graphs = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                smi = row.get(smiles_col, '').strip()
                if not smi: continue
                try:
                    g = smiles_to_graph(smi)
                    self.graphs.append(g)
                except ValueError:
                    continue

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        # 返回原子、键、边索引三元组
        return self.graphs[idx]
