# -*- coding: utf-8 -*-
import os
import ast
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from typing import List, Tuple, Dict

# ====================== JGR 单栏参数 ======================
FIGSIZE = (3.35, 5.0)   # 单栏常用宽度 ~3.35 in，高度按内容调
DPI     = 300
BASE_FONTSIZE = 8       # 单栏更紧凑的字号
CMAP    = "Greys"       # 色盲友好灰度
OUT_DIR = "figure"
OUT_BASENAME = "compare_singlecol"  # 导出名
LABELS_FALLBACK = ["EQ", "EP", "SS"]

mpl.rcParams.update({
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": BASE_FONTSIZE,
    "axes.labelsize": BASE_FONTSIZE,
    "xtick.labelsize": BASE_FONTSIZE,
    "ytick.labelsize": BASE_FONTSIZE,
    "legend.fontsize": BASE_FONTSIZE,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ====================== 数据读取与计算 ======================
def load_phase_dict(path: str) -> Dict:
    with open(path, "r") as f:
        return ast.literal_eval(f.read())

def infer_labels_by_index(phase_dict: Dict, C: int) -> List[str]:
    by_val = {}
    if isinstance(phase_dict, dict):
        for k, v in phase_dict.items():
            if isinstance(v, (int, np.integer)):
                by_val[int(v)] = str(k)
    labels = []
    for i in range(C):
        labels.append(by_val.get(i, LABELS_FALLBACK[i] if i < len(LABELS_FALLBACK) else f"Class {i}"))
    return labels

def load_data(file_name: str) -> Tuple[np.ndarray, float]:
    preds, trues, names = [], [], []
    with open(file_name, "r") as f:
        for line in f:
            s = line.strip().split(",")
            if len(s) < 3:
                continue
            preds.append(int(s[0])); trues.append(int(s[1])); names.append(s[2])

    preds = np.asarray(preds, dtype=int)
    trues = np.asarray(trues, dtype=int)
    names = np.asarray(names, dtype=object)

    uniq_events = np.unique(names)
    C = int(max(preds.max(initial=0), trues.max(initial=0))) + 1
    mat = np.zeros((C, C), dtype=np.int64)

    correct_flags = []
    for eid in uniq_events:
        mask = (names == eid)
        a1 = preds[mask]; a2 = trues[mask]
        c_pred = np.bincount(a1, minlength=C).argmax()
        c_true = np.bincount(a2, minlength=C).argmax()
        mat[c_true, c_pred] += 1
        correct_flags.append(c_pred == c_true)

    acc = float(np.mean(correct_flags)) if correct_flags else 0.0
    return mat, acc

def row_normalize(mat: np.ndarray) -> np.ndarray:
    denom = mat.sum(axis=1, keepdims=True)
    out = np.zeros_like(mat, dtype=float)
    ok = denom.squeeze(-1) > 0
    out[ok] = mat[ok] / denom[ok]
    return out

# ====================== 绘图 ======================
def annotate_panel_letter(ax, letter: str):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=BASE_FONTSIZE+1, fontweight="bold")

def plot_matrix(ax, mat_counts: np.ndarray, mat_percent: np.ndarray,
                class_labels: List[str], acc: float):
    C = mat_percent.shape[0]
    im = ax.imshow(mat_percent, cmap=CMAP, vmin=0.0, vmax=1.0,
                   origin="upper", aspect="equal", rasterized=True)

    # 文本标注（百分比 + 计数）
    for i in range(C):
        for j in range(C):
            val = float(mat_percent[i, j]); cnt = int(mat_counts[i, j])
            text = f"{val*100:.1f}%\n({cnt})"
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=BASE_FONTSIZE)

    ax.set_xticks(np.arange(C)); ax.set_yticks(np.arange(C))
    ax.set_xticklabels(class_labels); ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    # 细网格
    ax.set_xticks(np.arange(-0.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, C, 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5, alpha=0.15)
    ax.tick_params(axis="both", which="both", length=3)

    # 面板内微平均准确率
    ax.text(0.4, 0.02, f"Acc = {acc*100:.1f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=BASE_FONTSIZE)
    return im

def main():
    # 数据
    mat1, acc1 = load_data("odata/v9.orignal.txt")
    mat2, acc2 = load_data("odata/v9.transfer.txt")

    # 标签
    C = mat1.shape[0]
    phase_dict_path = "large/event.type"
    if os.path.exists(phase_dict_path):
        phase_dict = load_phase_dict(phase_dict_path)
        class_labels = infer_labels_by_index(phase_dict, C)
    else:
        class_labels = (LABELS_FALLBACK + [f"Class {i}" for i in range(C)])[:C]

    # 归一化
    mat1_percent = row_normalize(mat1)
    mat2_percent = row_normalize(mat2)

    # 图形（两行一列）
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
    gs = grid.GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.15)

    ax1 = fig.add_subplot(gs[0])
    im1 = plot_matrix(ax1, mat1, mat1_percent, class_labels, acc1)
    annotate_panel_letter(ax1, "(a)")

    ax2 = fig.add_subplot(gs[1])
    im2 = plot_matrix(ax2, mat2, mat2_percent, class_labels, acc2)
    annotate_panel_letter(ax2, "(b)")

    # 共享颜色条（右侧）
    # 仅使用最后一个 im（两者范围一致），并将 ax 传入列表做共享
    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized (%)")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([0, 25, 50, 75, 100])

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, OUT_BASENAME + ".png")
    svg_path = os.path.join(OUT_DIR, OUT_BASENAME + ".svg")
    pdf_path = os.path.join(OUT_DIR, OUT_BASENAME + ".pdf")
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    print(f"Saved: {png_path}\n       {svg_path}\n       {pdf_path}")

if __name__ == "__main__":
    main()
