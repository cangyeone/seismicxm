# -*- coding: utf-8 -*-
import os
import ast
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from typing import List, Tuple, Dict

# ====================== JGR 友好参数 ======================
FIGSIZE = (6.9, 4.8)   # 两排三列，适配双栏宽度
DPI     = 300
BASE_FONTSIZE = 9
CMAP    = "Greys"
OUT_DIR = "figure"
OUT_BASENAME = "compare_6panels"
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

def safe_load(path: str, C_hint: int | None = None) -> Tuple[np.ndarray, float]:
    """若文件不存在，返回零矩阵并提示，不中断绘图。"""
    if os.path.exists(path):
        return load_data(path)
    else:
        if C_hint is None: C_hint = 3
        print(f"[WARN] File not found: {path} -> using zeros.")
        return np.zeros((C_hint, C_hint), dtype=int), 0.0

# ====================== 绘图 ======================
def annotate_panel_letter(ax, letter: str):
    ax.text(0.02, 1.02, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=BASE_FONTSIZE+1, fontweight="bold")

def plot_matrix(ax, mat_counts: np.ndarray, mat_percent: np.ndarray,
                class_labels: List[str], acc: float):
    C = mat_percent.shape[0]
    im = ax.imshow(mat_percent, cmap=CMAP, vmin=0.0, vmax=1.0,
                   origin="upper", aspect="equal", rasterized=True)

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

    ax.set_xticks(np.arange(-0.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, C, 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5, alpha=0.15)
    ax.tick_params(axis="both", which="both", length=3)

    ax.text(0.65, 0.02, f"Acc = {acc*100:.1f}%", transform=ax.transAxes,
            ha="right", va="bottom", color="#888888", fontsize=BASE_FONTSIZE-2)
    return im

def main():
    steps = "200"

    # ---------- 测试集三面板 ----------
    f_test_orig =  f"odata/v9.orignal.test.{steps}.txt"
    f_test_tran = f"odata/v9.transfer.test.{steps}.txt"
    f_test_xgb  =  f"odata/v9.xgboost.txt"
    mat1, acc1 = load_data(f_test_xgb)
    mat2, acc2 = load_data(f_test_orig)
    mat3, acc3 = load_data(f_test_tran)
    

    # 类别标签
    C = mat1.shape[0]
    phase_dict_path = "large/event.type"
    if os.path.exists(phase_dict_path):
        phase_dict = load_phase_dict(phase_dict_path)
        class_labels = infer_labels_by_index(phase_dict, C)
    else:
        class_labels = (LABELS_FALLBACK + [f"Class {i}" for i in range(C)])[:C]

    # ---------- 训练集三面板（文件不存在则回退为零矩阵） ----------
    f_train_orig =  f"odata/v9.orignal.train.{steps}.txt"
    f_train_tran = f"odata/v9.transfer.train.{steps}.txt"
    f_train_xgb  =  f"odata/v9.xgboost.train.txt"

    mat4, acc4 = safe_load(f_train_xgb,  C_hint=C)
    mat5, acc5 = safe_load(f_train_orig, C_hint=C)
    mat6, acc6 = safe_load(f_train_tran, C_hint=C)
    

    # 行归一化
    mat1p, mat2p, mat3p = row_normalize(mat1), row_normalize(mat2), row_normalize(mat3)
    mat4p, mat5p, mat6p = row_normalize(mat4), row_normalize(mat5), row_normalize(mat6)

    # ---------- 作图：2×3 面板 + 底部共享 colorbar ----------
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
    gs = grid.GridSpec(2, 3, figure=fig, width_ratios=[1,1,1], height_ratios=[1,1],
                       wspace=0.02, hspace=0.02)

    # 上排（测试集）
    ax1 = fig.add_subplot(gs[0,0]); im1 = plot_matrix(ax1, mat1, mat1p, class_labels, acc1); annotate_panel_letter(ax1, "(A)")
    ax2 = fig.add_subplot(gs[0,1]); im2 = plot_matrix(ax2, mat2, mat2p, class_labels, acc2); annotate_panel_letter(ax2, "(B)")
    ax3 = fig.add_subplot(gs[0,2]); im3 = plot_matrix(ax3, mat3, mat3p, class_labels, acc3); annotate_panel_letter(ax3, "(C)")

    # 下排（训练集）
    ax4 = fig.add_subplot(gs[1,0]); im4 = plot_matrix(ax4, mat4, mat4p, class_labels, acc4); annotate_panel_letter(ax4, "(D)")
    ax5 = fig.add_subplot(gs[1,1]); im5 = plot_matrix(ax5, mat5, mat5p, class_labels, acc5); annotate_panel_letter(ax5, "(E)")
    ax6 = fig.add_subplot(gs[1,2]); im6 = plot_matrix(ax6, mat6, mat6p, class_labels, acc6); annotate_panel_letter(ax6, "(F)")

    # 底部横向 colorbar（共享全部面板）
    cbar = fig.colorbar(im6, ax=[ax1, ax2, ax3, ax4, ax5, ax6],
                        orientation="horizontal", location="bottom",
                        fraction=0.03, pad=0.10)
    cbar.set_label("Row-normalized (%)")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([0, 25, 50, 75, 100])

    # ---------- 保存 ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in (".png", ".svg", ".pdf"):
        fig.savefig(os.path.join(OUT_DIR, OUT_BASENAME + steps + ext), dpi=DPI)
    print("Saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
