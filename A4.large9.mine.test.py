from cmath import polar
import time
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import torch 
from models.EQLarge9 import EQLargeCNN as Model, Loss 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150

import torch.nn as nn

from torch.nn.modules.loss import CrossEntropyLoss, MSELoss 
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader 
from torch.nn.utils.rnn import pad_sequence
import os 
import numpy as np 
#from obspy.geodetics import degrees2kilometers, locations2degrees 
#import tqdm 
import scipy.interpolate as interpolate 
import h5py 



class MineDataset(Dataset):
    def __init__(self):
        f = torch.load("data/classification_data.data", weights_only=False)
        self.train_type0 = f["train0"]
        self.train_type1 = f["train1"]
        self.test_type0 = f["test0"]
        self.test_type1 = f["test1"]
        self.len0 = len(self.train_type0)
        self.len1 = len(self.train_type1)
        self.tests = self.test_type0+ self.test_type1

        self.tests = self.train_type0 + self.train_type1
        self.tests = self.tests[::5]
        print(len(self.train_type0), len(self.train_type1), len(self.test_type0), len(self.test_type1))
    def __len__(self):
        return len(self.tests)
    def __getitem__(self, idx):
     
        typeid, pidx, wave = self.tests[idx]
        #print(typeid)
        shift = np.random.randint(100, 300) 
        bidx = pidx - shift
        bidx = max(bidx, 0)
        x = wave[bidx:bidx+3072] 
        if len(x) < 3072:
            x = np.pad(x, ((0, 3072-len(x))))
        x /= np.max(np.abs(x)) + 1e-6 
        x = torch.tensor(x, dtype=torch.float32)
        x = torch.stack([x, x, x], dim=1)

        #print(x.shape, bidx, shift, wave.shape)
        info = torch.zeros([3072, 11]).to(x.dtype)
        #x = torch.cat([x, info], dim=1)
        m = torch.zeros([3072], dtype=torch.float32)
        m[shift:] = 1
        d = typeid
        
        #x = torch.tensor(x, dtype=torch.float32)
        # = torch.tensor(d, dtype=torch.long)
        #m = torch.tensor(m, dtype=torch.float32)
        return x, d, m 



def collate_batch(batch):
    """
    定义后处理函数
    """
    xs, ds, ms = [], [], []
    for x, d, m in batch:
        #print(x.shape, d.shape, m.shape)
        xs.append(x) 
        ds.append(d)
        ms.append(m)
    xs = torch.stack(xs, dim=0) 
    ds = torch.tensor(ds, dtype=torch.long) 
    ms = torch.stack(ms, dim=0)
    return xs, ds, ms

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.gridspec as grid 

def main(args):
    version = "transfer.balanced.200.mine.enc"#7.72
    model_name = f"testckpt/{version}.pt" #保存和加载神经网络模型权重的文件路径
    isb = ("balanced" in version)
    print(isb)
    train_dataset = MineDataset()     
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch, num_workers=1)
    device = torch.device("mps")
    model = Model()
    model.to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))


    trues = []
    preds = []
    preds1 = []
    preds2 = []
    preds3 = []
    waves = []
    if True:
        for x, d, m in train_dataloader:
            #print(x.shape, d.shape, m.shape)
            #print(d[:20, 0], m[0, :])
            x = torch.tensor(x, dtype=torch.float32, device=device) 
            d = torch.tensor(d, dtype=torch.long, device=device)
            x = x.to(device)
            d = d.to(device)
            #print(x.shape, d.shape)
            #print(inputs[0].shape)
            phase, polar, event_type, wave, mu, logvar, _ = model(x.permute(0, 2, 1))
            #print(o_etype.shape)
            d = d.detach().cpu().numpy()
            m = m.detach().cpu().numpy()
            v = event_type.argmax(dim=1).detach().cpu().numpy()
            trues.append(d)
            preds.append(v)
    # ================== 计算混淆矩阵 ==================
    trues = np.concatenate(trues)
    preds = np.concatenate(preds)

    cm = confusion_matrix(trues, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # 按行归一化，得到每类的召回率

    # 类别标签
    labels = ['Earthquake', 'Explosion']
    acc = np.mean(trues == preds) * 100

    # ================== 绘图美化 ==================
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)  # 去掉默认的 colorbar

    # 标题和坐标轴标签
    ax.set_title(f'Confusion Matrix  (Acc = {acc:.1f}%)', fontsize=11, pad=10)
    ax.set_xlabel('Predicted Category', fontsize=10)
    ax.set_ylabel('True Category', fontsize=10)

    # 坐标刻度字体
    ax.tick_params(axis='both', labelsize=9)

    # 保持图像为正方形
    ax.set_aspect('equal')

    # 在每个格子中写上“计数 / 百分比”
    # 百分比是相对该行（真实类别）的比例
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            perc = cm_norm[i, j] * 100
            text = f'({perc:.1f}%)'
            ax.text(
                j, i+0.15, text,
                ha='center', va='center',
                fontsize=7.5,
                color='white' if cm[i, j] > thresh else 'black'
            )

    plt.tight_layout()
    plt.savefig("figure/confusion_matrix.png")

#nohup python china.large6.py > logdir/large6.log 2>&1 &
#3583746
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拾取连续波形")          
    parser.add_argument('-d', '--dist', default=200, type=int, help="输入连续波形")       
    parser.add_argument('-o', '--output', default="result/t1", help="输出文件名")      
    parser.add_argument('-m', '--model', default="lppn.model", help="模型文件lppnmodel")                                                            
    args = parser.parse_args()      
    main(args)




