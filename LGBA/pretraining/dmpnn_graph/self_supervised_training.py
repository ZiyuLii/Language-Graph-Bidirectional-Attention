import os
os.environ['PYTORCH_NO_SHM'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'   # 一定在最上面


import torch.multiprocessing as _mp
_mp.set_sharing_strategy('file_system')

import time
import math
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
from multiprocessing import get_context

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lgba.pretraining.dmpnn_graph.smiles_graph_preprocess import GraphDataset
from lgba.pretraining.dmpnn_graph.graph_augmentation import augment_graph_pair
from lgba.pretraining.dmpnn_graph.dmpnn_encoder import DMPNNEncoder

cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch):
    return batch


def info_nce_loss(z1, z2, temperature):
    batch_size = z1.size(0)
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    sim = reps @ reps.T / temperature
    mask = (~torch.eye(2 * batch_size, device=device, dtype=torch.bool)).float()
    exp_sim = torch.exp(sim) * mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    pos = torch.cat([
        torch.diag(log_prob, batch_size),
        torch.diag(log_prob, -batch_size)
    ], dim=0)
    return -pos.mean()


class PrefetchLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None

    def __iter__(self):
        self.iter_loader = iter(self.loader)
        self.next_batch = None
        self._prefetch()
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        if self.stream:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self._prefetch()
        return batch

    def _prefetch(self):
        try:
            graphs = next(self.iter_loader)
        except StopIteration:
            self.next_batch = None
            return

        if self.device.type == 'cuda':
            with torch.cuda.stream(self.stream):
                g1_list, g2_list = [], []
                for graph in graphs:
                    (g1_cpu, _), (g2_cpu, _) = augment_graph_pair(graph)
                    g1_list.append((
                        g1_cpu.atom_features.cuda(self.device, non_blocking=True),
                        g1_cpu.bond_features.cuda(self.device, non_blocking=True),
                        g1_cpu.edge_index.cuda(self.device, non_blocking=True),
                    ))
                    g2_list.append((
                        g2_cpu.atom_features.cuda(self.device, non_blocking=True),
                        g2_cpu.bond_features.cuda(self.device, non_blocking=True),
                        g2_cpu.edge_index.cuda(self.device, non_blocking=True),
                    ))
            self.next_batch = (g1_list, g2_list)
        else:
            g1_list, g2_list = [], []
            for graph in graphs:
                (g1_cpu, _), (g2_cpu, _) = augment_graph_pair(graph)
                g1_list.append((g1_cpu.atom_features, g1_cpu.bond_features, g1_cpu.edge_index))
                g2_list.append((g2_cpu.atom_features, g2_cpu.bond_features, g2_cpu.edge_index))
            self.next_batch = (g1_list, g2_list)


def main():
    # 超参
    BATCH_SIZE   = 256
    EPOCHS       = 40
    LR           = 1e-3
    HIDDEN_DIM   = 128
    DEPTH        = 3
    TEMPERATURE  = 0.1
    PATIENCE     = 5
    LR_FACTOR    = 0.5
    LR_PATIENCE  = 2

    print("Device:", device)

    # 数据集 & DataLoader
    dataset = GraphDataset('ChemBL-LM_train.csv', smiles_col='SMILES')
    base_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        # multiprocessing_context=get_context('spawn')
    )
    loader = PrefetchLoader(base_loader, device)

    # 模型 & 优化器
    encoder = DMPNNEncoder(
        atom_feat_dim=dataset.graphs[0].atom_features.size(1),
        bond_feat_dim=dataset.graphs[0].bond_features.size(1),
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH
    ).to(device)
    proj_head = nn.Sequential(
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
    ).to(device)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=LR
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_FACTOR,
        patience=LR_PATIENCE
    )

    best_loss = float('inf')
    epochs_no_improve = 0

    os.makedirs('models', exist_ok=True)

    total_batches = math.ceil(len(dataset) / BATCH_SIZE)
    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        proj_head.train()

        total_loss = 0.0
        epoch_start = time.time()

        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        for batch_idx, (g1_batch, g2_batch) in enumerate(loader, start=1):
            # 前向、反向、更新
            h1 = torch.stack([encoder(a, b, e) for a, b, e in g1_batch], dim=0)
            h2 = torch.stack([encoder(a, b, e) for a, b, e in g2_batch], dim=0)
            z1, z2 = proj_head(h1), proj_head(h2)
            loss = info_nce_loss(z1, z2, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * h1.size(0)

            # 计算指标
            elapsed     = time.time() - epoch_start
            pct         = batch_idx / total_batches * 100
            avg_time    = elapsed / batch_idx
            remaining   = avg_time * (total_batches - batch_idx)

            # 打印简单日志，包含剩余时间
            print(
                f"Batch {batch_idx}/{total_batches} "
                f"({pct:5.1f}%)  "
                f"loss={loss.item():.4f}  "
                f"avg_time={avg_time:.2f}s  "
                f"elapsed={elapsed:.1f}s  "
                f"remaining={remaining:.1f}s"
            )

        # Epoch 完成，打印汇总
        epoch_time = time.time() - epoch_start
        avg_loss   = total_loss / len(dataset)
        print(f"--- Epoch {epoch} completed in {epoch_time:.1f}s, avg loss={avg_loss:.4f} ---")

        # 调度 & 早停
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(encoder.state_dict(), 'models/best_encoder.pth')
            torch.save(proj_head.state_dict(), 'models/best_proj_head.pth')
            print("Saved best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping after {epoch} epochs.")
                break

    print("\nTraining complete.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
