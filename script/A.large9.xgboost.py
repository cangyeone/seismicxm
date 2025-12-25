# -*- coding: utf-8 -*-
"""
XGBoost 三分类（地震/爆破/塌陷）示例
- 输入: X36 (N,36) numpy.ndarray，y (N,) numpy.ndarray, 标签取 {0,1,2}
- 主要步骤: 分层切分 -> (可选)网格搜索 -> 训练 -> 评估 -> 特征重要性
依赖: pip install xgboost scikit-learn numpy
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch 
# =========================
# 1) 准备你的数据
# =========================
# 假设你已经有以下变量：
# X36 = x36_feature   # shape: (N, 36)  dtype: float
# y   = x36_d         # shape: (N,)     dtype: int in {0,1,2}

data = torch.load("odata/feature.train.pth", weights_only=False)
X_train = np.array([item[0]["feature_36d"] for item in data], dtype=np.float32)
d_train = np.array([item[1] for item in data], dtype=np.int64)
e_train = [item[2] for item in data]

data = torch.load("odata/feature.test.pth", weights_only=False)
X_test = np.array([item[0]["feature_36d"] for item in data], dtype=np.float32)
d_test = np.array([item[1] for item in data], dtype=np.int64)
e_test = [item[2] for item in data]
# =========================
# 4) 基线模型（快速训练）
# =========================
base_model = XGBClassifier(
    objective="multi:softprob",  # 多分类概率
    num_class=3,
    n_estimators=400,            # 对应论文 t (迭代轮次)
    max_depth=3,                 # Dmax
    min_child_weight=8,          # Wmin
    learning_rate=0.2,           # eta
    gamma=0.2,                  # r
    subsample=0.7,               # S
    colsample_bytree=0.8,
    reg_lambda=5.0,
    tree_method="hist",          # 更快
    random_state=2025,
    eval_metric="mlogloss",
    n_jobs=-1
)

base_model.fit(X_train, d_train)
base_model.save_model("testckpt/xgb_model.json")
y_pred = base_model.predict(X_train)
print(y_pred.shape, X_train.shape)
acc = accuracy_score(y_pred, d_train)
print(f"[Baseline] Train Accuracy: {acc*100:.2f}%")
ofile = open("odata/v9.xgboost.train.ft.txt2", "w")
for a1, a2, a3 in zip(y_pred, d_train, e_train):
    ofile.write(f"{a1},{a2},{a3}\n")


y_test = base_model.predict(X_test)
acc = accuracy_score(y_test, d_test)
print(f"[Baseline] Test Accuracy: {acc*100:.2f}%")
ofile = open("odata/v9.xgboost.test.ft.txt2", "w")
for a1, a2, a3 in zip(y_test, d_test, e_test):
    ofile.write(f"{a1},{a2},{a3}\n")