#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

LABEL_FONT_SIZE = 13
VALUE_FONT_SIZE = 19
LABEL_SPREAD_FACTOR = 1.8
OUTPUT_FIG_W = 14.0
OUTPUT_FIG_H = 3.2
OUTPUT_DPI = 320
X_TICK_FONT_SIZE = 18
SHAP_GROUP_LABEL_FONT_SIZE = 18
SHAP_VALUE_FONT_SIZE = 22
SHAP_HIGHER_LOWER_FONT_SIZE = 22


def load_group_scores(csv_path: Path) -> List[Tuple[str, float, float]]:
    rows: List[Tuple[str, float, float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group_id = str(row["group_id"]).strip()
            sum_abs = float(row["sum_abs"])
            signed_sum = float(row["signed_sum"])
            rows.append((group_id, sum_abs, signed_sum))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _draw_gradient_projection(
    ax: plt.Axes,
    x0: float,
    x1: float,
    bar_y: float,
    label_y: float,
    color: str,
    alpha_top: float,
    alpha_bottom: float,
    expand_ratio: float,
    zorder: int,
) -> None:
    left = min(x0, x1)
    right = max(x0, x1)
    width = max(right - left, 1e-9)
    expand = width * float(expand_ratio)

    rgba = np.array(to_rgba(color), dtype=float)
    grad = np.ones((320, 4, 4), dtype=float)
    grad[..., :3] = rgba[:3]
    grad[..., 3] = np.linspace(alpha_top, alpha_bottom, 320)[:, None]

    img = ax.imshow(
        grad,
        extent=[left - expand, right + expand, label_y, bar_y],
        origin="lower",
        interpolation="bicubic",
        aspect="auto",
        zorder=zorder,
    )

    verts = np.array(
        [
            [left, bar_y],
            [right, bar_y],
            [right + expand, label_y],
            [left - expand, label_y],
        ]
    )
    clip = Polygon(verts, closed=True, facecolor="none", edgecolor="none", transform=ax.transData)
    ax.add_patch(clip)
    img.set_clip_path(clip)


def _draw_sign_projection(
    ax: plt.Axes,
    starts: List[float],
    ends: List[float],
    indices: List[int],
    bar_y: float,
    label_y: float,
    color: str,
    alpha_top: float,
    alpha_bottom: float,
    expand_ratio: float,
    zorder: int,
) -> None:
    if not indices:
        return
    left = min(min(starts[idx], ends[idx]) for idx in indices)
    right = max(max(starts[idx], ends[idx]) for idx in indices)
    _draw_gradient_projection(
        ax=ax,
        x0=float(left),
        x1=float(right),
        bar_y=bar_y,
        label_y=label_y,
        color=color,
        alpha_top=alpha_top,
        alpha_bottom=alpha_bottom,
        expand_ratio=expand_ratio,
        zorder=zorder,
    )


def _spread_label_centers(
    segments: List[Tuple[str, float, float, float]],
    indices: List[int],
    spread_factor: float,
) -> dict[int, float]:
    centers: dict[int, float] = {}
    if not indices:
        return centers
    raw_centers = {idx: 0.5 * (segments[idx][2] + segments[idx][3]) for idx in indices}
    if len(indices) == 1:
        only_idx = indices[0]
        centers[only_idx] = raw_centers[only_idx]
        return centers

    left_bound = min(segments[idx][2] for idx in indices)
    right_bound = max(segments[idx][3] for idx in indices)
    anchor = 0.5 * (left_bound + right_bound)
    span = (right_bound - left_bound) * float(spread_factor)
    target_left = anchor - 0.5 * span
    target_right = anchor + 0.5 * span
    target_positions = np.linspace(target_left, target_right, num=len(indices))
    ordered = sorted(indices, key=lambda idx: raw_centers[idx])
    for pos, idx in zip(target_positions, ordered):
        centers[idx] = float(pos)
    return centers


def _segment_polygon(x0: float, x1: float, y0: float, y1: float, tip: float, direction: str) -> np.ndarray:
    if abs(x1 - x0) < 1e-12:
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    tip = min(abs(tip), abs(x1 - x0) * 0.42)
    y_mid = 0.5 * (y0 + y1)
    if direction == "right":
        return np.array(
            [
                [x0, y0],
                [x1 - tip, y0],
                [x1, y_mid],
                [x1 - tip, y1],
                [x0, y1],
                [x0 + tip * 0.35, y_mid],
            ]
        )
    return np.array(
        [
            [x1, y0],
            [x0 + tip, y0],
            [x0, y_mid],
            [x0 + tip, y1],
            [x1, y1],
            [x1 - tip * 0.35, y_mid],
        ]
    )


def _top_value_label_names(rows: List[Tuple[str, float, float]], top_k: int = 5) -> set[str]:
    ranked = sorted(rows, key=lambda row: abs(row[2]), reverse=True)
    return {row[0] for row in ranked[: max(0, int(top_k))]}


def _plot_with_shap(
    rows: List[Tuple[str, float, float]],
    out_png: Path,
    base_value: float,
) -> bool:
    contribs = np.asarray([row[2] for row in rows], dtype=float)
    value_label_names = _top_value_label_names(rows, top_k=5)
    names = [
        f"{row[0]} = {row[2]:.2f}" if row[0] in value_label_names else row[0]
        for row in rows
    ]
    total_value = float(base_value + float(np.sum(contribs)))

    try:
        import shap  # type: ignore
    except Exception:
        return False

    try:
        plt.close("all")
        plt.figure(figsize=(OUTPUT_FIG_W, OUTPUT_FIG_H), dpi=OUTPUT_DPI)
        shap.plots.force(
            float(base_value),
            contribs,
            features=None,
            feature_names=names,
            matplotlib=True,
            show=False,
            figsize=(OUTPUT_FIG_W, OUTPUT_FIG_H),
            contribution_threshold=0.0,
        )

        fig = plt.gcf()
        ax = plt.gca()
        ax.set_position([0.035, 0.0001, 0.955, 0.62])

        for tick in ax.get_xticklabels():
            tick.set_fontsize(X_TICK_FONT_SIZE)

        for line in ax.lines:
            xdata = np.asarray(line.get_xdata(), dtype=float)
            ydata = np.asarray(line.get_ydata(), dtype=float)
            if len(xdata) == 2 and len(ydata) == 2 and abs(xdata[0] - xdata[1]) < 1e-8:
                line.set_linestyle("--")
                line.set_linewidth(1.0)
                line.set_alpha(0.75)

        target_texts = {"f(x)", f"{total_value:.2f}", "base value"}
        for txt in ax.texts:
            content = txt.get_text().strip()
            if content in target_texts:
                txt.set_visible(False)
            elif content in {"higher", "lower"}:
                txt.set_fontsize(SHAP_HIGHER_LOWER_FONT_SIZE)
            elif content.startswith("G"):
                txt.set_fontsize(SHAP_GROUP_LABEL_FONT_SIZE)

        ax.text(
            total_value,
            1.20,
            f"f(x) = {total_value:.2f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=SHAP_VALUE_FONT_SIZE,
            fontweight="bold",
            color="#2f3640",
        )

        fig.savefig(out_png, dpi=OUTPUT_DPI, facecolor="white")
        plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False


def plot_final_force_like(
    rows: List[Tuple[str, float, float]],
    out_png: Path,
    base_value: float = 0.0,
    top_label_n: int = 10,
    min_label_ratio: float = 0.08,
) -> Path:
    if len(rows) == 0:
        raise ValueError("No group data found in CSV.")

    if _plot_with_shap(rows=rows, out_png=out_png, base_value=base_value):
        return out_png

    total = float(base_value + sum(r[2] for r in rows))
    positives = [(name, sum_abs, signed) for name, sum_abs, signed in rows if signed > 0]
    negatives = [(name, sum_abs, signed) for name, sum_abs, signed in rows if signed < 0]

    positives.sort(key=lambda x: abs(x[2]), reverse=True)
    negatives.sort(key=lambda x: abs(x[2]), reverse=True)
    value_label_names = _top_value_label_names(rows, top_k=5)

    pos_total = float(sum(item[2] for item in positives))
    neg_total = float(sum(item[2] for item in negatives))
    left_extent = total - pos_total
    right_extent = total - neg_total

    pos_color = "#ff2b7a"
    neg_color = "#2d8cf0"
    axis_color = "#2f3640"
    helper_color = "#9aa3ad"

    x_min = min(left_extent, total, base_value)
    x_max = max(right_extent, total, base_value)
    x_span = max(x_max - x_min, 0.25)
    fig, ax = plt.subplots(figsize=(OUTPUT_FIG_W, OUTPUT_FIG_H), dpi=OUTPUT_DPI)
    ax.set_facecolor("white")

    y0 = -0.23
    y1 = -0.03
    tip = 0.018 * x_span
    top_label_y = y1 + 0.12
    bottom_label_y = y0 - 0.17

    pos_segments: List[Tuple[str, float, float, float]] = []
    cursor = float(left_extent)
    for name, _sum_abs, signed in positives:
        start = cursor
        end = cursor + float(signed)
        pos_segments.append((name, float(signed), start, end))
        cursor = end

    neg_segments: List[Tuple[str, float, float, float]] = []
    cursor = float(total)
    for name, _sum_abs, signed in negatives:
        start = cursor
        end = cursor + abs(float(signed))
        neg_segments.append((name, float(signed), start, end))
        cursor = end

    pos_label_indices = {idx for idx, item in enumerate(pos_segments) if item[0] in value_label_names}
    neg_label_indices = {idx for idx, item in enumerate(neg_segments) if item[0] in value_label_names}

    if pos_segments:
        _draw_sign_projection(
            ax=ax,
            starts=[seg[2] for seg in pos_segments],
            ends=[seg[3] for seg in pos_segments],
            indices=list(pos_label_indices),
            bar_y=y1,
            label_y=top_label_y - 0.10,
            color=pos_color,
            alpha_top=0.22,
            alpha_bottom=0.02,
            expand_ratio=0.10,
            zorder=1,
        )
    if neg_segments:
        _draw_sign_projection(
            ax=ax,
            starts=[seg[2] for seg in neg_segments],
            ends=[seg[3] for seg in neg_segments],
            indices=list(neg_label_indices),
            bar_y=y0,
            label_y=bottom_label_y - 0.06,
            color=neg_color,
            alpha_top=0.22,
            alpha_bottom=0.02,
            expand_ratio=0.10,
            zorder=1,
        )

    pos_label_centers = _spread_label_centers(pos_segments, sorted(pos_label_indices), LABEL_SPREAD_FACTOR)
    neg_label_centers = _spread_label_centers(neg_segments, sorted(neg_label_indices), LABEL_SPREAD_FACTOR)

    for idx, (name, value, start, end) in enumerate(pos_segments):
        poly = _segment_polygon(start, end, y0, y1, tip=tip, direction="right")
        ax.add_patch(Polygon(poly, closed=True, facecolor=pos_color, edgecolor="white", linewidth=1.1, zorder=3))
        if idx > 0:
            ax.plot([start, start + tip * 0.45], [y0, 0.5 * (y0 + y1)], color="white", linewidth=1.1, zorder=4)
            ax.plot([start, start + tip * 0.45], [y1, 0.5 * (y0 + y1)], color="white", linewidth=1.1, zorder=4)
        if idx in pos_label_indices:
            ax.text(
                pos_label_centers.get(idx, 0.5 * (start + end)),
                top_label_y,
                f"{name} = {value:.2f}",
                ha="center",
                va="bottom",
                fontsize=LABEL_FONT_SIZE,
                color=pos_color,
                clip_on=False,
                zorder=5,
            )

    for idx, (name, value, start, end) in enumerate(neg_segments):
        poly = _segment_polygon(start, end, y0, y1, tip=tip, direction="left")
        ax.add_patch(Polygon(poly, closed=True, facecolor=neg_color, edgecolor="white", linewidth=1.1, zorder=3))
        if idx > 0:
            ax.plot([start, start + tip * 0.45], [y0, 0.5 * (y0 + y1)], color="white", linewidth=1.1, zorder=4)
            ax.plot([start, start + tip * 0.45], [y1, 0.5 * (y0 + y1)], color="white", linewidth=1.1, zorder=4)
        if idx in neg_label_indices:
            ax.text(
                neg_label_centers.get(idx, 0.5 * (start + end)),
                bottom_label_y,
                f"{name} = {value:.2f}",
                ha="center",
                va="top",
                fontsize=LABEL_FONT_SIZE,
                color=neg_color,
                clip_on=False,
                zorder=5,
            )

    ax.axhline(0.0, color=axis_color, linewidth=1.0, zorder=0)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", which="both", top=True, labeltop=True, bottom=False, labelbottom=False, length=0)

    legend_x = 0.5 * (x_min + x_max)
    ax.text(legend_x - 0.015 * x_span, 0.10, "higher", ha="right", va="bottom", fontsize=10, color=pos_color, fontweight="bold")
    ax.text(legend_x, 0.10, "<->", ha="center", va="bottom", fontsize=10, color=helper_color)
    ax.text(legend_x + 0.015 * x_span, 0.10, "lower", ha="left", va="bottom", fontsize=10, color=neg_color, fontweight="bold")

    ax.text(total, 0.05, "f(x)", ha="center", va="bottom", fontsize=10, color=helper_color)
    ax.text(total, 0.00, f"{total:.2f}", ha="center", va="bottom", fontsize=VALUE_FONT_SIZE - 3, fontweight="bold", color=axis_color)
    ax.text(base_value, 0.10, "base value", ha="center", va="bottom", fontsize=10, color="#b8bec7")

    ax.set_xlim(-10.0, 10.0)
    ax.set_ylim(-0.62, 0.16)
    ax.set_yticks([])
    ticks = np.linspace(-10.0, 10.0, num=11)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:.1f}" for tick in ticks], fontsize=X_TICK_FONT_SIZE, color="#575f69")

    for spine in ("left", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["top"].set_linewidth(1.0)
    ax.spines["top"].set_color(axis_color)

    fig.subplots_adjust(left=0.03, right=0.995, top=0.88, bottom=0.18)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=OUTPUT_DPI, facecolor="white")
    plt.close(fig)
    return out_png


def resolve_group_csv(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.suffix.lower() == ".csv":
        return path
    if path.suffix.lower() == ".png" and path.name.endswith("_group_decision.png"):
        return path.with_name(path.name.replace("_group_decision.png", "_group_scores.csv"))
    raise ValueError("Input must be a *_group_scores.csv or *_group_decision.png path.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw final group contribution plot from an existing group decision output.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=r"explain_outputs\New\NF-kappaB_suppression_20260429_144920\06_Epicatechin_gallate\Epicatechin gallate_group_decision.png",
        # 输入改这行
        help="Path to *_group_decision.png or *_group_scores.csv",
    )
    parser.add_argument("--base-value", type=float, default=0.0, help="Base value used by the final contribution plot.")
    parser.add_argument("--top-label-n", type=int, default=8, help="Number of largest-magnitude groups to label.")
    parser.add_argument("--min-label-ratio", type=float, default=0.08, help="Minimum |contribution| / max(|contribution|) required for labeling.")
    parser.add_argument("--out", type=str, default="", help="Optional output PNG path.")
    args = parser.parse_args()

    group_csv = resolve_group_csv(args.input_path)
    rows = load_group_scores(group_csv)

    if args.out:
        out_png = Path(args.out)
    else:
        out_png = group_csv.with_name(group_csv.stem.replace("_group_scores", "_final_contribution_v4") + ".png")

    result = plot_final_force_like(
        rows=rows,
        out_png=out_png,
        base_value=args.base_value,
        top_label_n=args.top_label_n,
        min_label_ratio=args.min_label_ratio,
    )
    print(result)


if __name__ == "__main__":
    main()
