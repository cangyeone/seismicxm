# -*- coding: utf-8 -*-
import os
import ast
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from typing import List, Tuple, Dict

# ====================== JGR 友好参数 ======================
FIGSIZE = (8.0, 8.0)   # 适合双栏 ~7 in
DPI     = 300
BASE_FONTSIZE = 9      # JGR 推荐 8–12 pt
CMAP    = "Greys"      # 论文常用灰度（色盲友好）
OUT_DIR = "figure"
OUT_BASENAME = "compare.pnw.v2"  # compare.png / .svg / .pdf
LABELS_FALLBACK = ["EQ", "EP", "SS"]

# 字体与线宽（JGR/AGU 通常要求清晰、简洁）
mpl.rcParams.update({
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": BASE_FONTSIZE,   # 面板无需大标题，保持一致
    "axes.labelsize": BASE_FONTSIZE,
    "xtick.labelsize": BASE_FONTSIZE,
    "ytick.labelsize": BASE_FONTSIZE,
    "legend.fontsize": BASE_FONTSIZE,
    "axes.linewidth": 0.6,             # 轴线线宽 >= 0.5 pt
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ====================== 数据读取与计算 ======================
def load_phase_dict(path: str) -> Dict:
    """安全读取事件类型字典"""
    with open(path, "r") as f:
        return ast.literal_eval(f.read())

def infer_labels_by_index(phase_dict: Dict, C: int) -> List[str]:
    """
    按索引 0..C-1 从 phase_dict 获取标签名；
    支持 {"EQ":0,"EP":1,...}；若缺失则回退。
    """
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
    """
    输入文件：每行 'pred,true,event_name'
    对同一事件内样本采取众数投票（pred/true 分别取众数），
    生成 C×C 混淆矩阵（C=3）。返回矩阵与事件级 micro-accuracy。
    """
    preds, trues, names = [], [], []
    c = 0 
    with open(file_name, "r") as f:
        for line in f:
            s = line.strip().split(",")
            if len(s) < 3:
                continue
            if int(s[0]) > 2:continue 
            if int(s[1]) > 2:continue
            preds.append(int(s[0]))
            trues.append(int(s[1]))
            #names.append(s[2])
            names.append(c)
            c += 1 

    preds = np.asarray(preds, dtype=int)
    trues = np.asarray(trues, dtype=int)
    names = np.asarray(names, dtype=object)

    uniq_events = np.unique(names)
    # 自动推断类别数（保证与数据一致）
    C = int(max(preds.max(initial=0), trues.max(initial=0))) + 1
    C = 3 
    mat = np.zeros((C, C), dtype=np.int64)

    correct_flags = []
    for eid in uniq_events:
        mask = (names == eid)
        a1 = preds[mask]
        a2 = trues[mask]
        c_pred = np.bincount(a1, minlength=C).argmax()
        c_true = np.bincount(a2, minlength=C).argmax()
        mat[c_true, c_pred] += 1
        correct_flags.append(c_pred == c_true)

    acc = float(np.mean(correct_flags)) if correct_flags else 0.0
    return mat, acc

def row_normalize(mat: np.ndarray) -> np.ndarray:
    """
    行归一化（真实类为行），返回 0~1。全零行保持为零，避免 NaN。
    """
    denom = mat.sum(axis=1, keepdims=True)
    out = np.zeros_like(mat, dtype=float)
    ok = denom.squeeze(-1) > 0
    out[ok] = mat[ok] / denom[ok]
    return out

# ====================== 绘图 ======================
def annotate_panel_letter(ax, letter: str):
    """在面板左上角添加 A/B 标记（粗体）"""
    ax.text(0.02, 1.1, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=BASE_FONTSIZE+1, fontweight="bold")

def plot_matrix(ax, mat_counts: np.ndarray, mat_percent: np.ndarray,
                class_labels: List[str], acc: float):
    C = mat_percent.shape[0]
    im = ax.imshow(mat_percent, cmap=CMAP, vmin=0.0, vmax=1.0,
                   origin="upper", aspect="equal", rasterized=True)

    # 百分比 + 计数标注
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

    # 面板内准确率
    ax.text(0.5, 0.02, f"Acc = {acc*100:.1f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=BASE_FONTSIZE)
    return im
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
mpl.rcParams['axes.labelpad']   = 0     # 所有轴标题默认间距
mpl.rcParams['xtick.major.pad'] = 5      # x 方向刻度默认间距
mpl.rcParams['ytick.major.pad'] = 5
def main():
    # ---------- 混淆矩阵（事件级） ----------
    mat1, acc1 = load_data("odata/cross/orignal.200.pt.data1.txt")
    mat2, acc2 = load_data("odata/cross/transfer.200.pt.data1.txt")
    mat3, acc3 = load_data("odata/cross/orignal.200.pt.data2.txt")
    mat4, acc4 = load_data("odata/cross/transfer.200.pt.data2.txt")
    mat5, acc5 = load_data("odata/cross/orignal.pnw.200.pt.data1.txt")
    mat6, acc6 = load_data("odata/cross/transfer.pnw.200.pt.data1.txt")
    mat7, acc7 = load_data("odata/cross/orignal.pnw.200.pt.data2.txt")
    mat8, acc8 = load_data("odata/cross/transfer.pnw.200.pt.data2.txt")
    mat9, acc9 = load_data("odata/cross/orignal.pnw.balanced.200.pt.data1.txt")
    mat10, acc10 = load_data("odata/cross/transfer.pnw.balanced.200.pt.data1.txt")
    mat11, acc11 = load_data("odata/cross/orignal.pnw.balanced.200.pt.data2.txt")
    mat12, acc12 = load_data("odata/cross/transfer.pnw.balanced.200.pt.data2.txt")
    # ---------- 类别标签 ----------
    C = mat1.shape[0]
    phase_dict_path = "large/event.type"
    if os.path.exists(phase_dict_path):
        phase_dict = load_phase_dict(phase_dict_path)
        class_labels1 = infer_labels_by_index(phase_dict, C)
    else:
        class_labels1 = (LABELS_FALLBACK + [f"Class {i}" for i in range(C)])[:C]

    C = mat1.shape[0]
    phase_dict_path = "large/event.type"
    if os.path.exists(phase_dict_path):
        phase_dict = load_phase_dict(phase_dict_path)
        class_labels2 = infer_labels_by_index(phase_dict, C)
    else:
        class_labels2 = (LABELS_FALLBACK + [f"Class {i}" for i in range(C)])[:C]

    mats = [mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8, mat9, mat10, mat11, mat12]
    accs = [acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11, acc12] 
    print(class_labels1)
    print([m.shape for m in mats])
    # ---------- 作图（更小的子图间距 + 共享颜色条） ----------
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
    gs = grid.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.3)  # ← 缩小间距
    sbax1 = GridSpecFromSubplotSpec(
        3, 2, subplot_spec=gs[0], hspace=0.2, wspace=0.3
    )
    sbax2 = GridSpecFromSubplotSpec(
        3, 2, subplot_spec=gs[1], hspace=0.2, wspace=0.3
    )
    axs = []
    for i in range(6):
        mat1 = mats[i*2]
        mat2 = mats[i*2+1]
        acc1 = accs[i*2]
        acc2 = accs[i*2+1]
        # ---------- 归一化 ----------
        mat1_percent = row_normalize(mat1)
        mat2_percent = row_normalize(mat2)
        ax1 = fig.add_subplot(sbax1[i])
        im1 = plot_matrix(ax1, mat1, mat1_percent, class_labels1, acc1)
        annotate_panel_letter(ax1, "A")  # ← AB 标注
        axs.append(ax1)

        ax2 = fig.add_subplot(sbax2[i])
        im2 = plot_matrix(ax2, mat2, mat2_percent, class_labels2, acc2)
        annotate_panel_letter(ax2, "B")  # ← AB 标注
        axs.append(ax2)
    # 共享一个颜色条（放右侧，进一步减少中间空白）
    # 共享一个颜色条，放在底部横向
    cbar = fig.colorbar(im2, ax=axs,
                        orientation="horizontal",
                        location="bottom",
                        fraction=0.08, pad=0.12)
    cbar.set_label("Row-normalized (%)")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([0, 25, 50, 75, 100])

    # ---------- 保存 ----------
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
