import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec 
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec 
import scipy.signal as signal 
import os 
from matplotlib.ticker import FuncFormatter
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12
plt.switch_backend("agg") 
# -*- coding: utf-8 -*-
"""
读取 lppn-ctlg.txt 样式的数据，统计 P/S/Pn/Sn 的 TP/FP/FN、P/R/F1、
误差均值/标准差，并绘制每个相位的误差直方图。

主要修复与优化：
- 去掉 np.split：改为流式分块解析，更稳健更省内存。
- 避免可变默认参数：axs=None。
- 文件名解析修正：用 os.path.splitext。
- 增加健壮性：容错空行/缺字段/非法数值。
- 参数化：prob_min（概率阈值）、scale（单位缩放，默认 1/100）、bins、颜色等。
"""

import os
from typing import List, Optional, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 工具：安全转换
# ---------------------------
def _to_float(x: str, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(x[1:])

def _to_int(x: str, default: int = -9999) -> int:
    try:
        return int(float(x))  # 支持 "1.0" -> 1
    except Exception:
        return default

# ---------------------------
# 工具：按 #phase / #none 分块的生成器（流式，不占内存）
# 每个 block: List[List[str]]，第一行为 header
# ---------------------------
def iter_blocks(file_path: str):
    with open(file_path, "r") as f:
        block: List[List[str]] = []
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [s for s in line.split(",") if len(s) > 0]
            if not parts:
                continue
            # 碰到新的 header，就把旧 block 先 yield
            if parts[0].startswith("#") or parts[0].startswith("#none"):
                if block:
                    yield block
                    block = []
            block.append(parts)
        # 文件结束时还有残留 block
        if block:
            yield block

# ---------------------------
# 主函数：读取、统计并画图
# ---------------------------
def file_read(
    file_name: str = "lppn-ctlg.txt",
    gap1: float = 0.1,                 # 判定 TP 的误差阈值（秒）
    gap2: float = 0.5,                 # 进入统计直方图的误差上限（秒 & x 轴范围）
    axs: Optional[List[plt.Axes]] = None,
    mname: str = "",
    prob_min: Optional[float] = None,  # 最小置信度过滤（None 表示不过滤）
    scale: float = 1.0 / 100.0,        # 单位缩放：原始是样本点差/100 => 秒
    bins_main: int = 128,
    bins_sn: int = 6,
) -> Tuple[Optional[plt.Figure], List[plt.Axes], Dict[str, Dict[str, float]]]:
    """
    返回：
      fig, axes, metrics
      metrics 形如：
      {
        'P': {'P':0.95, 'S':..., 'Pn':..., 'Sn':...},
        'R': {...},
        'F1': {...},
        'mu_ms': {...},   # 误差均值（毫秒），ref - nn
        'sigma_ms': {...}
      }
    """
    phase_names = ["Pg", "Sg", "Pn", "Sn"]
    n_phase = 4

    # 统计容器
    tp = [0] * n_phase
    fp = [0] * n_phase
    fn = [0] * n_phase
    n_phases = [0, 0, 0, 0]
    # 误差（秒）的列表：ref - nn（用于直方图/均值/方差）
    err_lists: List[List[float]] = [[] for _ in range(n_phase)]

    # 逐块解析
    n_blocks = 0
    for block in iter_blocks(file_name):
        n_blocks += 1
        header = block[0]
        tag = header[0]  # "#phase" 或 "#none"

        # 参考时间（单位：原始单位；后续乘以 scale 转为秒）
        if tag.startswith("#none"):
            mm_time = [-1.0, -1.0, -1.0, -1.0]
        else:
            # header 形如: ["#phase", tP, tS, tPn, tSn, snr, ...]
            # 做好容错：长度不足时填 -1
            mm_time = []
            for i in range(1, 3):
                mm_time.append(_to_float(header[i], default=-1.0) if len(header) > i else -1.0)

        # allpick[i] = True 表示该相位“无需 FN”（因为无参考或已命中）
        allpick = [t < 0 for t in mm_time]
    
        # 遍历神经网络输出的每一条拾取
        for pidx, mm_t in enumerate(mm_time):
            if mm_t < 0:continue 
            is_detec = False 
            for row in block[1:]:
                if not row:
                    continue
                nn_type = _to_int(row[0], default=-1) - 1  # 0~3

                if nn_type < 0 or nn_type >= n_phase:
                    continue
                if pidx != nn_type:continue 
                nn_time = _to_float(row[1], default=np.nan)
                if np.isnan(nn_time):
                    continue

                nn_prob = _to_float(row[2], default=1.0) if len(row) >= 3 else 1.0
                #print(nn_prob)
                if (prob_min is not None) and (nn_prob < prob_min):
                    continue

                ref_time = mm_t 
                # 有参考时间
                if ref_time >= 0:
                    # 误差（秒）
                    err_abs = abs(ref_time - nn_time) * scale
                    if err_abs < gap1:
                        tp[nn_type] += 1
                        is_detec = True 
                    else:
                        fp[nn_type] += 1
                    # 进入误差分布（仅统计 |err| < gap2 的样本，和你原逻辑一致）
                    if err_abs < gap2:
                        err_lists[nn_type].append((ref_time - nn_time) * scale)
                else:
                    # 参考缺失仍被拾取 -> 视为 FP
                    fp[nn_type] += 1
            if not is_detec:
                # 无参考时间或无参考时间但未命中 -> 视为 FN
                fn[pidx] += 1
            ## 补未命中的 FN（有参考但最终没命中的）
            #for i in range(n_phase):
            #    if not allpick[i]:
            #        fn[i] += 1

    # 计算 P/R/F1、均值/方差（毫秒）
    eps = 1e-9
    P = []
    R = []
    F1 = []
    MU_ms = []
    SIGMA_ms = []
    for i in range(n_phase):
        pi = tp[i] / (tp[i] + fp[i] + eps)
        ri = tp[i] / (tp[i] + fn[i] + eps)
        fi = 2 * pi * ri / (pi + ri + eps)
        P.append(pi)
        R.append(ri)
        F1.append(fi)
        data = np.asarray(err_lists[i], dtype=float)
        MU_ms.append(float(np.mean(data) * 1e3) if data.size else np.nan)
        SIGMA_ms.append(float(np.std(data) * 1e3) if data.size else np.nan)

    #print(f"样本（块）数量: {n_blocks} | 文件: {file_name}")

    # 画图
    created_fig = False
    if axs is None:
        fig, axs = plt.subplots(1, n_phase, figsize=(14, 3.2), constrained_layout=True)
        created_fig = True
    else:
        fig = None
        if len(axs) < n_phase:
            raise ValueError(f"axs 的长度需要为 {n_phase}。")

    colors = ["#ff0000", "#0000ff", "#00aa00", "#888888"]
    base = os.path.splitext(os.path.basename(file_name))[0]

    for i in range(n_phase):
        ax = axs[i]
        dts = err_lists[i]
        if len(dts) == 0:
            ax.set_title(f"M:{mname}\nPhase:{phase_names[i]}\n无样本", fontsize=10, loc="left")
            ax.set_xlim([-gap2, gap2])
            ax.axvline(0.0, lw=1, ls="--")
            continue
        bins = bins_sn if phase_names[i] == "Sn" else bins_main
        ax.hist(dts, bins=bins, color=colors[i], alpha=0.5)
        # 标注
        pi, ri, fi = P[i], R[i], F1[i]
        mu_ms, sg_ms = MU_ms[i], SIGMA_ms[i]
        ax.set_title(
            f"M:{mname}\nPhase:{phase_names[i]}\n"
            f"P={pi:.3f}  R={ri:.3f}\nF1={fi:.3f}\n"
            fr"$\mu$={mu_ms:.3f} ms  $\sigma$={sg_ms:.3f} ms",
            fontsize=10, loc="left"
        )
        if i == 0:
            print(f"{mname} & {phase_names[i]} & {mu_ms:.3f} & {sg_ms:.3f} & {pi:.3f} & {ri:.3f} & {fi:.3f} \\\\")
        else:
            print(f"         & {phase_names[i]} & {mu_ms:.3f} & {sg_ms:.3f} & {pi:.3f} & {ri:.3f} & {fi:.3f} \\\\")
        ax.set_xlabel("Error [s]")
        ax.set_xlim([-gap2, gap2])
        ax.axvline(0.0, lw=1, ls="--")
        # y 轴使用科学计数法即可（数据量大时美观）
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

    metrics = {
        "P": {k: v for k, v in zip(phase_names, P)},
        "R": {k: v for k, v in zip(phase_names, R)},
        "F1": {k: v for k, v in zip(phase_names, F1)},
        "mu_ms": {k: v for k, v in zip(phase_names, MU_ms)},
        "sigma_ms": {k: v for k, v in zip(phase_names, SIGMA_ms)},
        "TP": {k: v for k, v in zip(phase_names, tp)},
        "FP": {k: v for k, v in zip(phase_names, fp)},
        "FN": {k: v for k, v in zip(phase_names, fn)},
        "blocks": n_blocks,
        "file": file_name,
        "model_name": mname,
    }

    # 可选：给整张图加一个总标题
    if created_fig and fig is not None:
        fig.suptitle(f"{base}  (gap1={gap1}s, gap2={gap2}s, prob_min={prob_min})", y=1.02, fontsize=11)

    return fig, axs, metrics

if __name__ == "__main__":
    gs = gridspec.GridSpec(4, 4, hspace=0.5)
    fig = plt.figure(1, figsize=(24, 24), dpi=100)
    #file_read("stdata/tinny16-2-ctlg.txt", 10000, "LPPN")
    axs = []

    for i in range(4):
        axt = []
        for j in range(4):
            #ax1, ax2 = fig.add_subplot(gs[i, j])#, fig.add_subplot(gs[i, j+2]) 
            ax1 = fig.add_subplot(gs[i, j])
            axt.append(ax1)
        axs.append(axt)
    gap1, gap2 = 1.0, 2.0 
    #file_read("stdata/stdt/eqt.pt.txt", gap1=gap1, gap2=gap2, axs=axs[0])
    
    #file_read("stdata/stdt/china.lppn2.pt.txt", gap1=gap1, gap2=gap2, axs=axs[0], mname="LPPN(M)")
    #file_read("stdata/stdt/china.unet.pt.txt", gap1=gap1, gap2=gap2, axs=axs[1], mname="UNet")
    #file_read("stdata/stdt/china.lppn3.pt.txt", gap1=gap1, gap2=gap2, axs=axs[2], mname="LPPN(T)")
    #file_read("stdata/stdt/china.unetpp.pt.txt", gap1=gap1, gap2=gap2, axs=axs[3], mname="UNet++")
    #file_read("stdata/stdt/china.eqt5.pt.txt", gap1=gap1, gap2=gap2, axs=axs[4], mname="EQT")
    #file_read("stdata/stdt/china.rnn.pt.txt", gap1=gap1, gap2=gap2, axs=axs[5], mname="RNN")
    #file_read("stdata/stdt/china.lppn6.pt.txt", gap1=gap1, gap2=gap2, axs=axs[6], mname="LPPN(L)")
    #file_read("stdata/stdt/china.bagging.txt", gap1=gap1, gap2=gap2, axs=axs[7], mname="Bagging")
    #[0.47, 0.51, 0.45, 0.45]
    #[0.30, 0.35, 0.20, 0.22]
    file_read("odata/seismicxm.pnw.txt",   gap1=gap1, gap2=gap2, axs=axs[0], mname="SeismicXM(0.2)",
               prob_min = 0.22)
    #[0.5, 0.39, 0.43, 0.1]
    #[0.41, 0.33, 0.28, 0.1]
    file_read("odata/phasenet.pnw.txt",   gap1=gap1, gap2=gap2, axs=axs[1], mname="PhaseNet",
               prob_min = 0.1)
    #[0.47, 0.31, 0.27, 0.1]
    #[0.30, 0.20, 0.20, 0.1]
    file_read("odata/seist.pnw.txt",   gap1=gap1, gap2=gap2, axs=axs[2], mname="SeisT",
               prob_min = 0.22)
    #[0.5, 0.39, 0.43, 0.1]
    #[0.41, 0.33, 0.28, 0.1]


    plt.savefig(f"figure/trans.pnw.jpg")
    plt.savefig(f"figure/trans.pnw.svg")
