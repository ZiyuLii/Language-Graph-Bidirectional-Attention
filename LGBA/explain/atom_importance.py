# atom_importance_heatmap.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 后端不弹窗
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from lgba.model.multimodal_regression_model import MultiModalRegressionModel
from lgba.pretraining.dmpnn_graph.smiles_graph_preprocess import smiles_to_graph


# ------------------------- 工具函数 -------------------------
def _sym_norm_list(x: List[float]) -> List[float]:
    if not x:
        return []
    m = max(abs(v) for v in x)
    if m < 1e-12:
        return [0.0 for _ in x]
    return [v / m for v in x]  # 映射到 [-1, 1]


def _ensure_2d_coords(mol: Chem.Mol) -> Chem.Mol:
    """确保 RDKit 分子对象具备 2D 坐标，便于可视化"""
    mol2d = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol2d)
    except Exception:
        pass
    rdDepictor.Compute2DCoords(mol2d)
    return mol2d


def _lerp_color(
    lo: Tuple[float, float, float],
    hi: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    """线性插值颜色"""
    t = float(np.clip(t, 0.0, 1.0))
    r = lo[0] + (hi[0] - lo[0]) * t
    g = lo[1] + (hi[1] - lo[1]) * t
    b = lo[2] + (hi[2] - lo[2]) * t
    return (float(r), float(g), float(b))


def _build_signed_atom_color_map(
    mol: Chem.Mol,
    atom_scores: List[float],
    pos_hi: Tuple[float, float, float] = (0.95, 0.15, 0.15),
    neg_hi: Tuple[float, float, float] = (0.20, 0.35, 0.95),
    base_color: Tuple[float, float, float] = (0.95, 0.95, 0.95),
    eps: float = 1e-12,
) -> Dict[int, Tuple[float, float, float]]:
    """将带符号的原子重要性映射为 RDKit 需要的颜色字典"""
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return {}

    arr = np.zeros(n_atoms, dtype=float)
    for i in range(n_atoms):
        if i < len(atom_scores):
            arr[i] = float(atom_scores[i])

    max_abs = float(np.max(np.abs(arr)))
    if max_abs < eps:
        return {i: base_color for i in range(n_atoms)}

    colors: Dict[int, Tuple[float, float, float]] = {}
    for idx in range(n_atoms):
        val = arr[idx]
        mag = abs(val) / max_abs
        if mag <= eps:
            colors[idx] = base_color
            continue
        target = pos_hi if val >= 0 else neg_hi
        colors[idx] = _lerp_color(base_color, target, mag)
    return colors


def _draw_atom_attention_image(
    mol: Chem.Mol,
    atom_scores: List[float],
    out_png: str,
    size: Tuple[int, int] = (680, 520),
    highlight_radius: float = 0.35,
    legend: Optional[str] = None,
) -> str:
    """使用 RDKit MolDraw2D 绘制原子注意力"""
    mol2d = _ensure_2d_coords(mol)
    colors = _build_signed_atom_color_map(mol2d, atom_scores)
    highlight_atoms = sorted(colors.keys())

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    try:
        # 新版本 RDKit 需要 DrawColour 对象
        opts.setBackgroundColour(rdMolDraw2D.DrawColour(1.0, 1.0, 1.0))
    except AttributeError:
        # 兼容旧版本（直接传元组）
        opts.setBackgroundColour((1.0, 1.0, 1.0))
    if hasattr(opts, "atomHighlightsAreCircles"):
        opts.atomHighlightsAreCircles = True
    if hasattr(opts, "circleHighlightRadius"):
        opts.circleHighlightRadius = float(highlight_radius)
    elif hasattr(opts, "highlightRadius"):
        opts.highlightRadius = float(highlight_radius)
    bw_palette = getattr(opts, "useBWAtomPalette", None)
    if callable(bw_palette):
        bw_palette()

    drawer.DrawMolecule(
        mol2d,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=colors,
    )

    if legend:
        draw_annot = getattr(drawer, "DrawAnnotation", None)
        if callable(draw_annot):
            align = getattr(rdMolDraw2D, "TextAlign_eRight", 0)
            draw_annot(0.98, 0.96, legend, 0.7, align=align)

    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    with open(out_png, 'wb') as f:
        f.write(png)
    return out_png


# ------------------------- 结果数据结构 -------------------------
@dataclass
class AtomExplainOutputs:
    atom_scores: List[float]                        # 融合后（带符号）的原子重要性
    atom_scores_components: Dict[str, List[float]]  # {'ig': [...], 'occ': [...]}
    fused_img_path: str
    ig_img_path: str
    occ_img_path: str
    bar_img_path: str

    # 兼容旧代码：mol_img_path 曾指向融合图
    @property
    def mol_img_path(self) -> str:  # type: ignore[override]
        return self.fused_img_path


# ------------------------- 解释器 -------------------------
class AtomExplainer:
    """
    原子级重要性解释器（红=正贡献，蓝=负贡献，RDKit 高亮圆圈）
    - IG：对 DMPNN 原子输入特征做 Integrated Gradients（保留符号）
    - OCC：特征遮挡（把某原子特征置零，相关边特征置零）
    - 融合：对称归一化后加权（默认 IG:OCC = 0.6:0.4）
    """
    def __init__(self, model: MultiModalRegressionModel, task: str = "regression",
                 device: Optional[torch.device] = None):
        self.model = model.eval()
        self.task = task  # 'regression' or 'classification'
        self.device = device if device is not None else getattr(model, "device", torch.device("cpu"))

    # ---------- 前向封装 ----------
    def _forward_from_graph(self, awd_vec: torch.Tensor,
                            atom_f: torch.Tensor, bond_f: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
        dmpnn_vec = self.model.dmpnn_encoder.encoder(atom_f, bond_f, edge_idx).unsqueeze(0)  # [1,128]
        fused = self.model.fusion(awd_vec, dmpnn_vec)                                        # [1,H]
        out = self.model.mlp(fused).squeeze()                                                # 标量
        return out

    # ---------- IG：原子特征集成梯度（带符号） ----------
    def _ig_atom_scores(self, smiles: str, steps: int = 64) -> Tuple[List[float], Chem.Mol]:
        g = smiles_to_graph(smiles)
        atom_f0 = g.atom_features.to(self.device)      # [N, Fa]
        bond_f  = g.bond_features.to(self.device)      # [E, Fb]
        edge_ix = g.edge_index.to(self.device)         # [2, E]

        with torch.no_grad():
            awd_vec = self.model.awd_encoder([smiles]).to(self.device)  # [1,400]

        baseline = torch.zeros_like(atom_f0)
        grads_acc = torch.zeros_like(atom_f0)

        for k in range(1, steps + 1):
            alpha = float(k) / steps
            atom_f = (baseline + alpha * (atom_f0 - baseline)).detach().requires_grad_(True)
            out = self._forward_from_graph(awd_vec, atom_f, bond_f, edge_ix)
            self.model.zero_grad(set_to_none=True)
            if atom_f.grad is not None:
                atom_f.grad.zero_()
            out.backward(retain_graph=True)
            grads_acc = grads_acc + atom_f.grad.detach()

        avg_grad = grads_acc / steps
        ig = (atom_f0 - baseline) * avg_grad                     # [N, Fa]
        atom_scores_signed = ig.sum(dim=1).detach().cpu().tolist()
        mol = Chem.MolFromSmiles(smiles)
        return atom_scores_signed, mol

    # ---------- OCC：原子级特征遮挡（带符号） ----------
    def _occ_atom_scores(
        self,
        smiles: str,
        occ_scale: float = 0.1,
        occ_affect_edges: bool = True,
    ) -> List[float]:
        g = smiles_to_graph(smiles)
        atom_f0 = g.atom_features.to(self.device)      # [N, Fa]
        bond_f0 = g.bond_features.to(self.device)      # [E, Fb]
        edge_ix = g.edge_index.to(self.device)         # [2, E]
        N = atom_f0.size(0)
        src, dst = edge_ix[0], edge_ix[1]

        with torch.no_grad():
            awd_vec = self.model.awd_encoder([smiles]).to(self.device)
            base = self._forward_from_graph(awd_vec, atom_f0, bond_f0, edge_ix)
            y0 = float(base.item())
            if self.task == "classification":
                y0 = float(torch.sigmoid(torch.tensor(y0)).item())

        scores = []
        for i in range(N):
            atom_f = atom_f0.clone()
            bond_f = bond_f0.clone()
            # 局部遮挡：原子 i 特征置零 + 与 i 相连的边特征置零
            atom_f[i, :] = atom_f[i, :] * float(occ_scale)
            mask_edges = ((src == i) | (dst == i))
            if occ_affect_edges and mask_edges.any():
                bond_f[mask_edges] = bond_f[mask_edges] * float(occ_scale)

            out = self._forward_from_graph(awd_vec, atom_f, bond_f, edge_ix)
            y_mask = float(out.item())
            if self.task == "classification":
                y_mask = float(torch.sigmoid(torch.tensor(y_mask)).item())
            scores.append(y0 - y_mask)  # 正：遮挡降低输出 → 该原子正贡献；负：相反
        return scores

    # ---------- 原子 Top‑k 条形图（带正负色） ----------
    def _plot_topk_atoms(self, atom_scores: List[float], k: int, out_png: str) -> str:
        idx_sorted = sorted(range(len(atom_scores)), key=lambda i: abs(atom_scores[i]), reverse=True)[:k]
        labels = [f"Atom#{i}" for i in idx_sorted]
        vals   = [atom_scores[i] for i in idx_sorted]
        colors = ['#F14040' if v >= 0 else '#1A6FDF' for v in vals]


        plt.figure(figsize=(6, 4), dpi=200)
        plt.barh(labels, vals, color=colors)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        return out_png

    # ---------- 主接口 ----------
    def explain_smiles(
        self,
        smiles: str,
        steps: int = 64,
        k: int = 12,
        out_prefix: str = "explain",
        sigma: float = 0.35,
        res: int = 480,
        occ_scale: float = 0.1,
        occ_affect_edges: bool = True,
        out_dir: str = "explain_outputs",
    ) -> AtomExplainOutputs:
        os.makedirs(out_dir, exist_ok=True)

        # 1) IG（带符号）
        ig_atoms, mol = self._ig_atom_scores(smiles, steps=steps)

        # 2) OCC（带符号）
        occ_atoms = self._occ_atom_scores(
            smiles,
            occ_scale=occ_scale,
            occ_affect_edges=occ_affect_edges,
        )

        # 3) 融合：对称归一化后加权
        ig_n  = _sym_norm_list(ig_atoms)
        occ_n = _sym_norm_list(occ_atoms)
        merged = [0.6*ig + 0.4*oc for ig, oc in zip(ig_n, occ_n)]

        # 4) 三张 RDKit 高亮图 + 原子 Top‑k
        fused_img = os.path.join(out_dir, f"{out_prefix}_atoms_fused.png")
        ig_img    = os.path.join(out_dir, f"{out_prefix}_atoms_ig.png")
        occ_img   = os.path.join(out_dir, f"{out_prefix}_atoms_occ.png")
        # sigma/res 参数保留兼容：sigma -> 高亮圆半径，res -> 画布边长
        canvas_side = max(int(res), 200) if res else 480
        highlight_radius = max(float(sigma), 0.05)
        size = (canvas_side, canvas_side)
        _draw_atom_attention_image(mol, merged, fused_img, size=size, highlight_radius=highlight_radius, legend="Fused (0.6*IG + 0.4*OCC)")
        _draw_atom_attention_image(mol, ig_n,   ig_img,  size=size, highlight_radius=highlight_radius, legend="Integrated Gradients")
        _draw_atom_attention_image(mol, occ_n,  occ_img, size=size, highlight_radius=highlight_radius, legend="Occlusion")
        bar_img = os.path.join(out_dir, f"{out_prefix}_atoms_topk.png")
        self._plot_topk_atoms(merged, k=k, out_png=bar_img)

        return AtomExplainOutputs(
            atom_scores=merged,
            atom_scores_components={"ig": ig_n, "occ": occ_n},
            fused_img_path=fused_img,
            ig_img_path=ig_img,
            occ_img_path=occ_img,
            bar_img_path=bar_img
        )


# ------------------------- 用法示例 -------------------------
if __name__ == "__main__":
    # 按需修改以下路径
    import torch
    model = MultiModalRegressionModel(
        spe_vocab_path="models/vocab-spe.pkl",
        awd_vocab_path="models/vocab-awd.pkl",
        awd_model_path="models/smiles_encoder.pth",
        dmpnn_model_path="models/best_encoder.pth",
        device=torch.device("cpu")
    )
    # 可选：加载已训练权重（回归示例）
    ckpt = "AID_743287_test1_best_classifier.pth"
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        state = {k: v for k, v in state.items() if k in model.state_dict()}
        model.load_state_dict(state, strict=False)
        print("Loaded trained weights.")

    model.eval()
    explainer = AtomExplainer(model, task="classification")
    smi = "C1[C@H]([C@H](OC2=C1C(=CC(=C2[C@@H]3[C@H]([C@H](OC4=CC(=CC(=C34)O)O)C5=CC(=C(C=C5)O)O)OC(=O)C6=CC(=C(C(=C6)O)O)O)O)O)C7=CC(=C(C=C7)O)O)O"  # 阿司匹林
    out = explainer.explain_smiles(smi, out_prefix="Procyanidin B2 3-O-gallate", sigma=0.35, res=560)
    print("融合原子分数前10:", out.atom_scores[:10])
    print("图像：", out.fused_img_path, out.ig_img_path, out.occ_img_path, out.bar_img_path)
