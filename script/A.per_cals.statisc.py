# -*- coding: utf-8 -*-
import os
import ast
import numpy as np
from typing import List, Tuple, Dict

LABELS_FALLBACK = ["EQ", "EP", "SS"]

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
    """
    读取预测结果文件：
    每行: pred,true,event_id
    先在事件内部做多数投票 -> 事件级混淆矩阵 + 事件级准确率
    """
    preds, trues, names = [], [], []
    with open(file_name, "r") as f:
        for line in f:
            s = line.strip().split(",")
            if len(s) < 3:
                continue
            preds.append(int(s[0]))
            trues.append(int(s[1]))
            names.append(s[2])

    preds = np.asarray(preds, dtype=int)
    trues = np.asarray(trues, dtype=int)
    names = np.asarray(names, dtype=object)

    uniq_events = np.unique(names)
    C = int(max(preds.max(initial=0), trues.max(initial=0))) + 1
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

def safe_load(path: str, C_hint: int | None = None) -> Tuple[np.ndarray, float]:
    """若文件不存在，返回零矩阵并提示。"""
    if os.path.exists(path):
        return load_data(path)
    else:
        if C_hint is None:
            C_hint = 3
        print(f"[WARN] File not found: {path} -> using zeros.")
        return np.zeros((C_hint, C_hint), dtype=int), 0.0

# ====================== 统计指标 ======================
def compute_per_class_metrics(mat: np.ndarray, class_labels: List[str]) -> Dict[str, Dict[str, float]]:
    """
    根据混淆矩阵计算每一类的 Precision / Recall / F1 / support，
    以及 macro / weighted 平均。
    mat[i, j] = true=i, pred=j
    """
    C = mat.shape[0]
    metrics: Dict[str, Dict[str, float]] = {}

    total_samples = mat.sum()
    supports = mat.sum(axis=1)  # 每一行 = 该类真实数量

    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0

    for k in range(C):
        label = class_labels[k] if k < len(class_labels) else f"Class {k}"
        TP = mat[k, k]
        FN = mat[k, :].sum() - TP
        FP = mat[:, k].sum() - TP
        support = supports[k]

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics[label] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": float(support),
        }

        macro_p += prec
        macro_r += rec
        macro_f1 += f1

        weight = support / total_samples if total_samples > 0 else 0.0
        weighted_p += prec * weight
        weighted_r += rec * weight
        weighted_f1 += f1 * weight

    if C > 0:
        macro_p /= C
        macro_r /= C
        macro_f1 /= C

    metrics["macro_avg"] = {
        "precision": macro_p,
        "recall": macro_r,
        "f1": macro_f1,
        "support": float(total_samples),
    }
    metrics["weighted_avg"] = {
        "precision": weighted_p,
        "recall": weighted_r,
        "f1": weighted_f1,
        "support": float(total_samples),
    }

    return metrics

def print_metrics(name: str, mat: np.ndarray, acc: float, class_labels: List[str]):
    print("=" * 70)
    print(f"Model / File: {name}")
    print(f"Overall accuracy (event-level): {acc*100:.2f}%")
    metrics = compute_per_class_metrics(mat, class_labels)

    print("- Per-class metrics:")
    for lbl in class_labels:
        m = metrics.get(lbl, None)
        if m is None:
            continue
        print(
            f"  {lbl:>3s} | "
            f"support = {int(m['support']):4d} | "
            f"P = {m['precision']*100:6.2f}% | "
            f"R = {m['recall']*100:6.2f}% | "
            f"F1 = {m['f1']*100:6.2f}%"
        )

    m_macro = metrics["macro_avg"]
    m_weighted = metrics["weighted_avg"]
    print("- Macro average: "
          f"P = {m_macro['precision']*100:6.2f}%, "
          f"R = {m_macro['recall']*100:6.2f}%, "
          f"F1 = {m_macro['f1']*100:6.2f}%")
    print("- Weighted average: "
          f"P = {m_weighted['precision']*100:6.2f}%, "
          f"R = {m_weighted['recall']*100:6.2f}%, "
          f"F1 = {m_weighted['f1']*100:6.2f}%")
    print()

# ====================== 主函数 ======================
def main():
    # ---------- 测试集三文件 ----------
    f_test_orig = "odata/v9.orignal.txt"
    f_test_tran = "odata/v9.transfer.txt"
    f_test_xgb  = "odata/v9.xgboost.txt"

    mat_xgb,  acc_xgb  = load_data(f_test_xgb)
    mat_orig, acc_orig = load_data(f_test_orig)
    mat_tran, acc_tran = load_data(f_test_tran)

    # 类别标签
    C = mat_xgb.shape[0]
    phase_dict_path = "large/event.type"
    if os.path.exists(phase_dict_path):
        phase_dict = load_phase_dict(phase_dict_path)
        class_labels = infer_labels_by_index(phase_dict, C)
    else:
        class_labels = (LABELS_FALLBACK + [f"Class {i}" for i in range(C)])[:C]

    # 只对测试集统计分类指标（你可以根据需要加上训练集）
    print_metrics("XGBoost (test)",  mat_xgb,  acc_xgb,  class_labels)
    print_metrics("Original model (test)", mat_orig, acc_orig, class_labels)
    print_metrics("Transfer model (test)", mat_tran, acc_tran, class_labels)

    # ---------- 如果你也想看训练集，可以解开下面注释 ----------
    # f_train_orig = "odata/v9.orignal.train.txt"
    # f_train_tran = "odata/v9.transfer.train.txt"
    # f_train_xgb  = "odata/v9.xgboost.train.txt"
    #
    # mat4, acc4 = safe_load(f_train_xgb,  C_hint=C)
    # mat5, acc5 = safe_load(f_train_orig, C_hint=C)
    # mat6, acc6 = safe_load(f_train_tran, C_hint=C)
    #
    # print_metrics("XGBoost (train)",  mat4, acc4, class_labels)
    # print_metrics("Original model (train)", mat5, acc5, class_labels)
    # print_metrics("Transfer model (train)", mat6, acc6, class_labels)

if __name__ == "__main__":
    main()
