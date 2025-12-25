# -*- coding: utf-8 -*-
"""
SRL-friendly confusion grid of waveforms:
- 3x3 subplots: rows = True [EQ, EP, SS], cols = Pred [EQ, EP, SS]
- Up to MAX_TRACES_PER_CELL waveforms stacked per cell (robust when <5)
- Grayscale-friendly, 9pt font, double-column figure size
"""

import time
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# === 项目依赖 ===
from utils.forplt import LargeDataTest2
from models.EQLarge9 import EQLargeCNN as Model, Loss

# === SRL/期刊友好参数（可按需修改） ===
FIGSIZE_INCH = (7.0, 5.6)   # 双栏友好 ~7 in 宽
DPI = 300
BASE_FONTSIZE = 9
LINEWIDTH = 0.5
TRACE_SPACING = 1.25         # 波形堆叠间距
MAX_TRACES_PER_CELL = 5      # 每个格子最多绘制的波形条数
TIME_LABEL = "Time (s)"
Y_LABEL = "Normalized amplitude"
OUT_DIR = "figure"
OUT_BASENAME = "mat.wave"    # 将导出 mat.wave.png / .pdf / .svg
SAMPLE_RATE = 100.0          # 采样率Hz（根据你的数据改动）
CMAP_COLOR = "k"             # 灰度友好：黑色曲线

# === Matplotlib 全局样式 ===
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": BASE_FONTSIZE,
    "axes.linewidth": 0.8,
    "axes.titlesize": BASE_FONTSIZE,
    "axes.labelsize": BASE_FONTSIZE,
    "xtick.labelsize": BASE_FONTSIZE - 1,
    "ytick.labelsize": BASE_FONTSIZE - 1,
    "legend.fontsize": BASE_FONTSIZE - 1,
    "pdf.fonttype": 42,   # 使 PDF 文本可编辑
    "ps.fonttype": 42,
})

TYPES = ["EQ", "EP", "SS"]  # 0,1,2 的顺序映射

def _panel_letters(nr=3, nc=3):
    letters = []
    base = ord('a')
    for i in range(nr * nc):
        letters.append(f"{chr(base + i)})")
    return np.array(letters).reshape(nr, nc)

def _normalize_trace(w):
    w = np.asarray(w, dtype=np.float32)
    w = w - np.nanmean(w)
    mx = np.nanmax(np.abs(w))
    if not np.isfinite(mx) or mx < 1e-6:
        return np.zeros_like(w, dtype=np.float32)
    return w / mx

@torch.no_grad()
def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    # ====== 可配置参数 ======
    version = "04b"
    steps = "100"
    model_name = f"testckpt/transfer.{steps}.pt"  # 模型权重
    phase_name_file = "large/event.type"          # 若需映射，可用；此处不强依赖
    batch_size = 5                                # 期望批大小，允许返回不足
    n_batches = 176                               # 循环批次数（按需调整）

    # ====== 加载数据与模型 ======
    device = torch.device("cpu")
    dtype = torch.float32
    model = Model().to(device).eval()
    state = torch.load(model_name, map_location=device)
    model.load_state_dict(state)

    data_tool = LargeDataTest2("test")

    # （可选）读取标签映射，不强制使用
    id2p = {0: "EQ", 1: "EP", 2: "SS"}
    if os.path.exists(phase_name_file):
        try:
            with open(phase_name_file, "r") as f:
                phase_dict = eval(f.read())
                # 文件若为 name->id，则翻转以得到 id->name
                if isinstance(phase_dict, dict):
                    tmp = {}
                    for k, v in phase_dict.items():
                        tmp[v] = k
                    id2p.update(tmp)
        except Exception:
            pass

    # ====== 画布与网格 ======
    fig = plt.figure(figsize=FIGSIZE_INCH)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.10)
    axs = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]
    letters = _panel_letters(3, 3)

    # 轴标题与角标
    for i in range(3):
        for j in range(3):
            ax = axs[i][j]
            # 角标
            ax.text(0.9, 1.02, letters[i, j], transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=BASE_FONTSIZE,
                    fontweight="bold")
            # 列标题
            if i == 0:
                ax.set_title(f"Pred: {TYPES[j]}")
            # 行标签
            if j == 0:
                ax.set_ylabel(f"True: {TYPES[i]}")

            # 统一外观
            #ax.spines["top"].set_visible(False)
            #ax.spines["right"].set_visible(False)
            ax.tick_params(which="both", direction="out", length=3)

    # 统计每格子已绘制条数
    counts = np.zeros((3, 3), dtype=int)

    # ====== 逐批推理与绘图 ======
    t_global_max = 0.0  # 用于统一 x 轴范围
    for bi in range(n_batches):
        # 允许返回小于 batch_size 的样本
        batch = data_tool.batch_data(batch_size)
        # 兼容：有的工具返回三元组，有的返回更多，这里仅取前三项
        try:
            x, d, c = batch[:3]
        except Exception:
            x, d, c = batch

        # 转 tensor
        x = torch.as_tensor(x, dtype=dtype, device=device)            # [B, T, C]
        d = torch.as_tensor(d, dtype=torch.long, device=device).view(-1)  # [B]

        # 前向
        # 模型接口：phase, polar, event_type, wave, mu, logvar, ccc = model(x.permute(0,2,1))
        out = model(x.permute(0, 2, 1))
        if isinstance(out, (list, tuple)) and len(out) >= 3:
            event_type = out[2]
        else:
            raise RuntimeError("Model forward output format unexpected.")

        p = torch.argmax(event_type, dim=1).detach().cpu().numpy()
        d = d.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()  # [B, T, C]

        B = x_np.shape[0]
        T = x_np.shape[1]
        t = np.arange(T, dtype=np.float32) / float(SAMPLE_RATE)
        t_global_max = max(t_global_max, float(t[-1]))

        # 遍历样本
        for k in range(B):
            true_id = int(d[k]) if 0 <= int(d[k]) < 3 else 0
            pred_id = int(p[k]) if 0 <= int(p[k]) < 3 else 0

            # 拿 Z 分量（索引2）；若无则退回最后一列
            z_idx = 2 if x_np.shape[-1] >= 3 else (x_np.shape[-1] - 1)
            w = _normalize_trace(x_np[k, :, z_idx])

            # 当前格子已达上限则跳过
            if counts[true_id, pred_id] >= MAX_TRACES_PER_CELL:
                continue

            ax = axs[true_id][pred_id]
            offset = counts[true_id, pred_id] * TRACE_SPACING
            ax.plot(t, w + offset, lw=LINEWIDTH, c=CMAP_COLOR, clip_on=True)
            counts[true_id, pred_id] += 1

        # 可选：打印进度
        # print(f"STEP {bi}")

    # ====== 统一轴范围与标签 ======
    # y 轴范围按各格子 max count 自适应
    for i in range(3):
        for j in range(3):
            ax = axs[i][j]
            n_here = max(1, counts[i, j])
            ymin = -1.05
            ymax = (n_here - 1) * TRACE_SPACING + 1.05
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(0.0, max(0.01, t_global_max))

            # 仅最左列显示 y 轴标签与刻度
            if j != 0:
                ax.set_yticklabels([])
                ax.set_yticks([])

            # 仅最下行显示 x 轴标签，其余隐藏刻度标签
            if i == 2:
                ax.set_xlabel(TIME_LABEL)
            else:
                ax.set_xticklabels([])

    # 顶层公共 y 轴标签（更清晰）
    fig.text(0.04, 0.5, Y_LABEL, va="center", rotation="vertical", fontsize=BASE_FONTSIZE)

    # 布局与导出
    fig.tight_layout(rect=[0.06, 0.04, 1.00, 0.98])

    png_path = os.path.join(OUT_DIR, f"{OUT_BASENAME}.png")
    pdf_path = os.path.join(OUT_DIR, f"{OUT_BASENAME}.pdf")
    svg_path = os.path.join(OUT_DIR, f"{OUT_BASENAME}.svg")

    fig.savefig(png_path)
    fig.savefig(pdf_path)  # 期刊友好（矢量）
    fig.savefig(svg_path)

    # 简短统计输出（可选）
    total_drawn = int(counts.sum())
    print(f"绘制完成：共 {total_drawn} 条波形；每格子计数：\n{counts}")
    print(f"已保存：\n  {png_path}\n  {pdf_path}\n  {svg_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SRL-friendly waveform confusion grid")
    parser.add_argument('-d', '--dist', default=200, type=int, help="输入连续波形（未直接使用，占位）")
    parser.add_argument('-o', '--output', default="result/t1", help="输出文件名（未直接使用，占位）")
    parser.add_argument('-m', '--model', default="lppn.model", help="模型文件（未直接使用，占位）")
    args = parser.parse_args()
    main(args)
