
# -*- coding: utf-8 -*-
import os
import ast
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from typing import List, Tuple, Dict

# ====================== JGR 友好参数 ======================
FIGSIZE = (9.0, 7.0)   # 适合双栏 ~7 in
DPI     = 300
BASE_FONTSIZE = 9      # JGR 推荐 8–12 pt
CMAP    = "Greys"      # 论文常用灰度（色盲友好）
OUT_DIR = "figure"
OUT_BASENAME = "compare.pnw"  # compare.png / .svg / .pdf
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


    mat_percent = row_normalize(mat)

    accs = []
    for i in range(C):
        if mat[i, i] == 0 and i == 2:
            continue
        accs.append(mat_percent[i, i])
    acc = np.mean(accs)

    return mat, acc
import matplotlib.patheffects as path_effects
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
    ax.text(-0.01, 1.01, letter, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=BASE_FONTSIZE+1, fontweight="bold")

def plot_matrix(ax, mat_counts: np.ndarray, mat_percent: np.ndarray,
                class_labels: List[str], acc: float):
    C = mat_percent.shape[0]
    im = ax.imshow(mat_percent, cmap=CMAP, vmin=0.0, vmax=1.0,
                   origin="upper", aspect="equal", rasterized=True)

    # 百分比 + 计数标注
    accs = []
    for i in range(C):
        for j in range(C):
            val = float(mat_percent[i, j]); cnt = int(mat_counts[i, j])
            text = f"{val*100:.1f}%\n({cnt})"
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=6)
            if i == j:
                if val > 1e-6 and i == 2:
                    accs.append(val)
    #acc = np.mean(accs) if accs else 0.0
    ax.set_xticks(np.arange(C)); ax.set_yticks(np.arange(C))
    ax.set_xticklabels(class_labels); ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    # 细网格
    ax.set_xticks(np.arange(-0.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, C, 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5, alpha=0.15)
    ax.tick_params(axis="both", which="both", length=3)

    # 面板内准确率
    #ax.text(0.5, 0.02, f"Acc = {acc*100:.1f}%", transform=ax.transAxes,
    #        ha="right", va="bottom", fontsize=BASE_FONTSIZE)
    txt = ax.text(0.5, 0.01, f"mean_acc = {acc*100:.1f}%",
                  transform=ax.transAxes,
                  ha="center", va="bottom",
                  fontsize=4,
                  fontweight="bold",
                  color="white")  # 主体白色文字
    # 添加黑色描边
    txt.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='black'),
        path_effects.Normal()
    ])
    return im
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
mpl.rcParams['axes.labelpad']   = 0     # 所有轴标题默认间距
mpl.rcParams['xtick.major.pad'] = 5      # x 方向刻度默认间距
mpl.rcParams['ytick.major.pad'] = 5

from dataclasses import dataclass

@dataclass
class RunItem:
    path: str
    regime: str       # 'xgb' | 'original' | 'transfer'
    train_src: str    # 'IM' | 'PNW' | 'PNW_bal'
    testset: str      # 'data1' | 'data2'

def parse_run(path: str) -> RunItem:
    name = os.path.basename(path).lower()
    # 训练策略/模型
    if name.startswith("xgb"):
        regime = "xgb"
    elif "transfer" in name:
        regime = "transfer"
    else:
        regime = "original"  # 兼容 orignal/ original

    # 训练数据来源
    if ".pnw.balanced." in name or "pnw.balanced" in name:
        train_src = "PNW_bal"
    elif ".pnw." in name or "pnw." in name:
        train_src = "PNW"
    else:
        train_src = "IM"

    # 测试集
    testset = "data2" if "data2" in name else "data1"
    return RunItem(path, regime, train_src, testset)
def hide_yaxis(ax):
    ax.set_ylabel("")                                 # 去掉 y 轴标题
    ax.tick_params(axis="y", which="both",
                   left=False, right=False,
                   labelleft=False, labelright=False) # 去掉刻度与文字
    ax.spines["left"].set_visible(False)              # 隐藏左侧轴线

def main():
    # ========== 结果文件 ==========
    # XGBoost（新增，放在列表最前；相对路径按你存放处调整）
    xgb_files = [
        "odata/cross/xgb.data1.txt",
        "odata/cross/xgb.data2.txt",
        "odata/cross/xgb.pnw.data1.txt",
        "odata/cross/xgb.pnw.data2.txt",
        "odata/cross/xgb.pnw.balanced.data1.txt",
        "odata/cross/xgb.pnw.balanced.data2.txt",
    ]

    # 原来的 12 个（original/transfer × IM/PNW/PNW_bal × data1/2）
    ot_files = [
        "odata/cross/orignal.200.pt.data1.txt",
        "odata/cross/transfer.200.pt.data1.txt",
        "odata/cross/orignal.200.pt.data2.txt",
        "odata/cross/transfer.200.pt.data2.txt",
        "odata/cross/orignal.pnw.200.pt.data1.txt",
        "odata/cross/transfer.pnw.200.pt.data1.txt",
        "odata/cross/orignal.pnw.200.pt.data2.txt",
        "odata/cross/transfer.pnw.200.pt.data2.txt",
        "odata/cross/orignal.pnw.balanced.200.pt.data1.txt",
        "odata/cross/transfer.pnw.balanced.200.pt.data1.txt",
        "odata/cross/orignal.pnw.balanced.200.pt.data2.txt",
        "odata/cross/transfer.pnw.balanced.200.pt.data2.txt",
    ]

    # 合并（XGB 在最前）
    file_list = xgb_files + ot_files
    runs = [parse_run(p) for p in file_list]

    # ---------- 类别标签 ----------
    mat_tmp, _ = load_data(runs[0].path)
    C = mat_tmp.shape[0]
    phase_dict_path = "large/event.type"
    if os.path.exists(phase_dict_path):
        phase_dict = load_phase_dict(phase_dict_path)
        class_labels = infer_labels_by_index(phase_dict, C)
    else:
        class_labels = (LABELS_FALLBACK + [f"Class {i}" for i in range(C)])[:C]

    # ---------- 目标顺序：三大列 × 三行 × 两小图 ----------
    regime_order = ["xgb", "original", "transfer"]   # XGBoost 放最左
    train_order  = ["IM", "PNW", "PNW_bal"]          # 上到下
    test_order   = ["data1", "data2"]                # 左到右

    # 索引表： (train_src, regime, testset) -> path
    lookup = {(r.train_src, r.regime, r.testset): r.path for r in runs}

    # ---------- 作图 ----------
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
    gs_outer = grid.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.2)

    # 为三列分别创建 3×2 的子网格
    blocks = [
        GridSpecFromSubplotSpec(3, 2, subplot_spec=gs_outer[i], hspace=0.1, wspace=0.1)
        for i in range(3)
    ]
    col_titles = [
        "XGBoost (tree-based)",
        "Trained from scratch",
        "Fine-tuned",
    ]

    axs, last_im = [], None
    panel_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 2  # 足够用
    letter_idx = 0

    # 逐列绘制
    for col, regime in enumerate(regime_order):
        # 列标题
        title_ax = fig.add_subplot(grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_outer[col])[0])
        title_ax.axis("off")
        title_ax.text(0.5, 1.0, col_titles[col],
                      ha="center", va="top",
                      fontsize=BASE_FONTSIZE+1, fontweight="bold",
                      transform=title_ax.transAxes)

        block = blocks[col]
        # 三行（训练来源）
        for r, train_src in enumerate(train_order):
            # 两小图（测试集）
            for c, testset in enumerate(test_order):
                ax = fig.add_subplot(block[r, c])
                path = lookup.get((train_src, regime, testset))

                if not path or not os.path.exists(path):
                    ax.text(0.5, 0.5, "missing", ha="center", va="center")
                    ax.set_xticks([]); ax.set_yticks([])
                    annotate_panel_letter(ax, panel_letters[letter_idx]); letter_idx += 1
                    axs.append(ax)
                    continue

                mat_counts, acc = load_data(path)
                mat_percent = row_normalize(mat_counts)
                last_im = plot_matrix(ax, mat_counts, mat_percent, class_labels, acc)
                annotate_panel_letter(ax, panel_letters[letter_idx]); letter_idx += 1
                axs.append(ax)
                if c == 1:
                    hide_yaxis(ax)
                # 小标题：训练来源 → 测试集

                st = "PNW-B" if train_src == "PNW_bal" else train_src
                if testset == "data1":
                    testset = "IM"
                else:
                    testset = "PNW"
                ax.set_title(f"{st} → {testset}", pad=4)

    # 共享颜色条（底部）
    cbar = fig.colorbar(last_im, ax=axs, orientation="horizontal",
                        location="bottom", fraction=0.02, pad=0.12)
    cbar.set_label("Row-normalized (%)")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([0, 25, 50, 75, 100])

    # ---------- 图注（SRL 简洁说明） ----------
    caption = (
        "Confusion matrices (event-level, row-normalized). "
        "Training data: IM (Inner Mongolia), PNW, and PNW (balanced). "
        "Test data: data1, data2. "
        "Models: XGBoost (left), Original from scratch (middle), Transfer fine-tuned (right)."
    )
    # 置于底部（constrained_layout 下用 figure.text 较稳）
    #fig.text(0.5, -0.015, caption, ha="center", va="top", fontsize=BASE_FONTSIZE)

    # ---------- 保存 ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, OUT_BASENAME + ".png")
    svg_path = os.path.join(OUT_DIR, OUT_BASENAME + ".svg")
    pdf_path = os.path.join(OUT_DIR, OUT_BASENAME + ".pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}\n       {svg_path}\n       {pdf_path}")

main()