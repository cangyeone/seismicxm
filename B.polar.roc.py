# -*- coding: utf-8 -*-
"""
整洁的 ROC 绘图脚本（JGR/AGU 友好风格）
- 多个模型对比：读取预生成的 batch 测试数据（data/testdata/testwave.{step}.pt）
- 正确的 ROC: x=FPR, y=TPR；AUC 使用 sklearn.metrics.auc
- 稳健性：缺失文件跳过、设备自动选择、无梯度推理
- 图形风格：高分辨率、清晰字体、细网格、紧凑布局
"""

import os
import time
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import roc_curve, auc

# 你的模型
from models.EQLarge9 import EQLargeCNNPolar as Model, Loss  # noqa: F401  # Loss 未用到

# ====================== 图形/版式参数（期刊友好） ======================
FIGSIZE = (3.35, 3.35)   # 单栏常用 ~3.35 in
DPI = 300
BASE_FONTSIZE = 9
LINEWIDTH = 1.6
MARKERSIZE = 3.0
LEGEND_FONTSIZE = 8

mpl.rcParams.update({
    "figure.figsize": FIGSIZE,
    "figure.dpi": DPI,
    "font.size": 8,              # 整体字号调小
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,        # 图例更小
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

plt.switch_backend("agg")  # 后端设为无界面

# ====================== 配置 ======================
TEST_PATTERN = "data/testdata/testwave.{step}.pt"  # 每个 step 对应一个 .pt
N_STEPS = 200                                      # 测试 step 数
OUT_DIR = "figure"
OUT_BASENAME = "roc"                               # 输出名：roc.png / roc.pdf
POSITIVE_CLASS = 0                                 # 设定哪个类别为“正类”，与你的标签一致

MODEL_PATHS = [
    "ckpt/out2.pt",
    "ckpt/enc2.pt",
    "ckpt/tra2.pt",
    "ckpt/all2.pt",
    "ckpt/newtrain.all2",
]
MODEL_NAMES = [
    "Dec",
    "Enc+Dec",
    "Tr+Dec",
    "All",
    "NewTrain",
]


def pick_device():
    """自动选择推理设备：优先 MPS，再 CUDA，否则 CPU。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_batches(n_steps, positive_class=0):
    """
    读取 N_STEPS 批数据，拼接为整体数组。
    返回：
        y_true: (N,) int64 真实标签（取 a2 的第 0 列）
        waves:  (N, C, T) float32 波形（转置过）
    """
    ys, waves = [], []
    missing = 0
    for step in range(n_steps):
        path = TEST_PATTERN.format(step=step)
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            f = torch.load(path, weights_only=False)  # PyTorch>=2.6 兼容
        except Exception as e:
            print(f"[WARN] 读取 {path} 失败：{e}")
            missing += 1
            continue

        a1, a2 = f["a1"], f["a2"]  # a1: (B, T, C), a2: (B, ?)
        # 统一到 (B, C, T)
        w = torch.tensor(a1, dtype=torch.float32).permute(0, 2, 1).numpy()
        y = np.array(a2)[:, 0]  # 使用标签的第 0 列

        waves.append(w)
        ys.append(y)

    if missing > 0:
        print(f"[INFO] 跳过 {missing} 个缺失/损坏的 step 文件")

    if len(ys) == 0:
        raise RuntimeError("没有可用的测试数据，请检查 TEST_PATTERN 与 N_STEPS")

    waves = np.concatenate(waves, axis=0)  # (N, C, T)
    y_true = np.concatenate(ys, axis=0).astype(np.int64)  # (N,)
    # 断言标签合理
    if not np.isin(positive_class, y_true).any():
        print(f"[WARN] 设定的正类 {positive_class} 在标签中未出现，请确认标签含义")
    return y_true, waves


@torch.inference_mode()
def model_scores(model_path, device, waves, positive_class=0):
    """
    对某个模型计算所有样本的“正类”概率评分。
    参数：
        model_path: 模型权重路径
        device:     推理设备
        waves:      (N, C, T) numpy float32
    返回：
        y_score:    (N,) float64 正类的概率
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model = Model()
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device).eval()

    batch = 2048  # 可按显存/内存调整
    N = waves.shape[0]
    scores = np.empty((N,), dtype=np.float64)

    for i in range(0, N, batch):
        j = min(N, i + batch)
        x = torch.from_numpy(waves[i:j]).to(device)  # (B, C, T)
        y1, y2 = model(x)  # y1: logits for type 分类
        p = torch.softmax(y1, dim=1)[:, positive_class]  # 正类概率
        scores[i:j] = p.detach().cpu().numpy()

    return scores


def plot_rocs(results):
    """
    绘制多模型 ROC，对 results 列表中每项绘图。
    results: List[ (label, fpr, tpr, auc_value) ]
    """
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])

    # 随机分类器参考线
    xx = np.linspace(0, 1, 200)
    ax.plot(xx, xx, linestyle="--", linewidth=1.0, color="0.4", label="Random")

    for label, fpr, tpr, auc_value in results:
        ax.plot(
            fpr,
            tpr,
            linewidth=LINEWIDTH,
            label=f"{label}, AUC={auc_value:.3f}",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.grid(True, which="both", ls=":", lw=0.6, color="0.8")
    ax.legend(
        loc="lower right",       # 右下角
        frameon=False,           # 去掉边框
        handlelength=2.0,
        handletextpad=0.5,
    )

    png_path = os.path.join(OUT_DIR, f"{OUT_BASENAME}.png")
    pdf_path = os.path.join(OUT_DIR, f"{OUT_BASENAME}.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[OK] 保存图像：{png_path} / {pdf_path}")



EXTRA_FILE = "data/polar_extra.txt"   # 你要绘制的额外文件路径

def load_extra_scores(path, pos_class=0):
    """
    读取额外文件，格式: label, p0, p1
    返回:
        y_true: (N,)
        y_score: (N,) 属于 pos_class 的概率
    """
    data = np.loadtxt(path, delimiter=",")
    y_true = data[:, 0].astype(int)
    y_score = data[:, pos_class+1]   # 第2列是类0概率, 第3列是类1概率
    return y_true, y_score

def main():
    t0 = time.time()
    device = pick_device()
    print(f"[INFO] 使用设备：{device}")

    # 载入并拼接所有测试样本
    y_true, waves = load_batches(N_STEPS, positive_class=POSITIVE_CLASS)

    results = []

    # ============ 加载并绘制额外文件 ROC =============


    for mp, name in zip(MODEL_PATHS, MODEL_NAMES):
        try:
            print(f"[INFO] 推理模型：{name}  ({mp})")
            scores = model_scores(mp, device, waves, positive_class=POSITIVE_CLASS)
            # sklearn 自动为我们扫阈值，得到 fpr, tpr
            fpr, tpr, _ = roc_curve(
                y_true == POSITIVE_CLASS,
                scores,
                pos_label=True,
                drop_intermediate=True,
            )
            auc_value = auc(fpr, tpr)
            results.append((name, fpr, tpr, auc_value))
            print(f"    AUC = {auc_value:.4f}")
        except FileNotFoundError:
            print(f"[WARN] 模型文件不存在：{mp}，已跳过")
        except Exception as e:
            print(f"[WARN] 模型 {name} 计算失败：{e}")

    # ============ 加载并绘制额外文件 ROC =============
    if os.path.exists(EXTRA_FILE):
        y_true, y_score = load_extra_scores(EXTRA_FILE, pos_class=POSITIVE_CLASS)
        fpr, tpr, _ = roc_curve(y_true == POSITIVE_CLASS, y_score)
        auc_value = auc(fpr, tpr)
        results.append(("Oringal Model", fpr, tpr, auc_value))
        print(f"[INFO] 外部文件 ROC: AUC={auc_value:.3f}")
    else:
        print(f"[WARN] 外部文件 {EXTRA_FILE} 不存在，跳过。")
    
    res = [results[-1]]
    for r in results[:-1]:
        res.append(r)
    #results = [r for r in results[:-1]]

    if len(results) == 0:
        raise RuntimeError("没有可绘制的曲线（全部模型失败或缺失）。")

    plot_rocs(res)
    print(f"[DONE] 总耗时：{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
