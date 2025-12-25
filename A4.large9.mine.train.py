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
        print(len(self.train_type0), len(self.train_type1), len(self.test_type0), len(self.test_type1))
        print(len(self.train_type0) + len(self.test_type0), len(self.train_type1)+len(self.test_type1))
    def __len__(self):
        return len(self.train_type0) * 5 + len(self.train_type0) * 5
    def __getitem__(self, idx):
        if idx % 2 == 0:
            typeid, pidx, wave = self.train_type0[idx%self.len0] 
        else:
            typeid, pidx, wave = self.train_type1[idx%self.len1]
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
def main(args):
    version = "transfer.balanced.200.mine.enc"#7.72
    model_name = f"testckpt/{version}.pt" #保存和加载神经网络模型权重的文件路径
    isb = ("balanced" in version)
    print(isb)
    train_dataset = MineDataset()     
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch, num_workers=1)
    device = torch.device("mps")
    model = Model()
    try:
        model.load_state_dict(torch.load(model_name, map_location=device))
    except:
        print("模型不存在！")
        pass 
    isfixed = True
    if isfixed:
        for key, var in model.named_parameters():
            if var.dtype != torch.float32:continue # BN统计计数无梯度
            if "decoder_event_type" in key or "emb" in key: # 仅有最后一层有out
                var.requires_grad = True
            else:
                var.requires_grad = False  
    model.to(device)
    model.train()
    lossfn = Loss() 
    lossfn.to(device)
    acc_time = 0 #记录训练的累计时间
    outloss = open(f"logdir/largev9.{version}.txt", "w") #记录训练过程中的loss
    if isfixed:
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 1e-4, weight_decay=1e-4)
    else:
        optim = torch.optim.Adam(model.parameters(), 1e-2, weight_decay=1e-3)
    
    phase_name_file = "large/phase.type"
    with open(phase_name_file, "r") as f:
        phase_dict = eval(f.read())
        id2p = {0:"Noise"} 
        for k, v in phase_dict.items():
            id2p[v+1] = k 
    lossfn = nn.CrossEntropyLoss()
    step = 0 
    for e in range(1):
        for x, d, m in train_dataloader:
            st = time.perf_counter()
            x = torch.tensor(x, dtype=torch.float32, device=device) 
            d = torch.tensor(d, dtype=torch.long, device=device)
           
            #print(x.shape, d.shape)
            #print(inputs[0].shape)
            phase, polar, event_type, wave, mu, logvar, _ = model(x.permute(0, 2, 1))

            #print(oc.shape, op.shape, ot.shape, d.shape, d1.shape, m1.shape, tp.shape)
            loss = lossfn(event_type, d) 
            loss.backward()
            if loss.isnan():
                print("NAN error")
                optim.zero_grad()
                continue 
            optim.step() 
            optim.zero_grad()
            ls = loss.detach().cpu().numpy()
            ed = time.perf_counter()
            if step % 10 == 0:
                torch.save(model.state_dict(), model_name)
                p = torch.argmax(event_type, dim=1).detach().cpu().numpy() 
                d = d.detach().cpu().numpy()
                print(f"{step},{e},{ed - st},{ls},{np.mean(p==d)}\n")
                outloss.write(f"{step},{ed - st},{ls}\n")
                outloss.flush()
                acc_time += ed - st
            step += 1 
        torch.save(model.state_dict(), model_name)
    print("done!")
    print("done!")
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




