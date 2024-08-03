from cmath import polar
import time
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import torch 
from utils.large7 import DataLarge9
from models.EQLarge9b import EQLargeCNN as Model
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150
import torch 
import torch.nn as nn 
import torch.nn.functional as F

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.l1 = nn.L1Loss()
    def forward(self, x, d, x1, d1, m1, x2, d2, ow, wave, mu, logvar, eloc, ploc):
        m = d[:, 1:, :].clone()
        # 1 2 4 8
        m[:, 0, :] *= 2
        m[:, 1, :] *= 2
        m[:, 2, :] *= 4 
        m[:, 3, :] *= 8 
        m = torch.sum(m, dim=1)
        B, C, T = x.shape 
        loss = (d * torch.log(x+1e-6)).sum(dim=1) * (m * 1 + 1) # 加权加大
        loss0 = - (loss).sum() / B /10 * 100
        #loss0 = - (d * torch.log(x+1e-9)).sum() / B#震相
        loss1 = self.ce(x1, d1) * m1 
        loss1 = loss1.sum() * 10 / B 
        #loss1 = loss1.sum()  * 80
    
        loss2 = self.ce(x2, d2) 
        loss2 = loss2.sum() * 100  * B/256
        #loss2 = loss2.sum() * 10240
        loss31 = (torch.square(ow - wave)).sum() / B /10
        loss32 = torch.mean(logvar**2) * 1e-2
        loss3 = loss31 + loss32 

        d = eloc 
        y = ploc[:, :3]
        loss4a = torch.sum((eloc-ploc[:, :3]))
        loss4b = torch.sum((torch.sqrt(torch.sum(y**2, dim=1)) - torch.sqrt(torch.sum(d**2, dim=1)))**2)
        loss4 = loss4a + loss4b 

        loss0 = loss0 * 0.1   # phase 
        loss1 = loss1 * 1     # polar 
        loss2 = loss2 * 10     # type  
        loss3 = loss3 * 100   # wave 
        loss4 = loss4 * 0.001 # location

        loss = loss0  + loss1 + loss2 + loss3 + loss4
        #print(loss0,loss1,loss2)
        return loss, loss0, loss1, loss2, loss3, loss4
def main(args):
    version = "03.step1"
    model_name = f"ckpt_large/largev9b.{version}.pt" #保存和加载神经网络模型权重的文件路径, ckpt_large/largev9o.03.step1.pt
    #ckpt_large/largev8.02.step3.pt
    data_tool = DataLarge9(file_name="h5data", stride=1, n_length=10240, padlen=2048, std=0.2, max_dist=2000)
    #data_tool = DataPnSnWithPolarType(file_name="h5data", stride=1, n_length=20480, padlen=4096)
    device = torch.device("cuda:1")
    model = Model()
    try:
        model.load_state_dict(torch.load(model_name, map_location=device))
    except:
        print("模型未加载")
        #for par in model.parameters():
        #    if len(par.shape)>=2:
        #        torch.nn.init.kaiming_normal_(par)
        pass
    model.to(device)
    model.train()
    lossfn = Loss() 
    lossfn.to(device)
    acc_time = 0 #记录训练的累计时间
    outloss = open(f"logdir/loss/large/largev9b.{version}.txt", "a") #记录训练过程中的loss
    optim = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-2)
    #optim = torch.optim.SGD(model.parameters(), 1e-5, weight_decay=1e-3)
    for step in range(50000000):
        st = time.perf_counter() #记录当前时间，用于计算每个训练步骤的执行时间
        a1, a2, a3, b1, b2, b3, tp, eloc = data_tool.batch_data(batch_size=128) #获取一个批次的地震波数据 #128  32
        #print(a1.shape)
        #print(a3.shape)
        #a1.shape (128, 20480, 3)
        #a3.shape (128, 20480, 5)
        #N = len(a1) 
        B, T, C = a1.shape 
        #for k in range(B):
        #    if np.random.random()>0.5:continue 
        #    t = np.arange(T) * 0.01 
        #    m = (np.sin(t * np.pi * 2 * np.random.uniform(5, 30))>0.0).astype(np.float64) 
        #    a1[k, :] = m * a1[k, :]  
        wave = torch.tensor(a1, dtype=torch.float).to(device) 
        wave = wave.permute(0, 2, 1)
        d = torch.tensor(a3, dtype=torch.float32).to(device) 
        d = d.permute(0, 2, 1)
        eloc = torch.tensor(eloc, dtype=torch.float32).to(device) 
        d1 = torch.tensor(b1, device=device, dtype=torch.long) 
        #print(d1.shape) torch.Size([128, 10240])
        polartype = torch.tensor(b1, dtype=torch.int)
        
        polartype = polartype[0]
        #poalr[N, 2, 2048]
        #pidx = 1000
        #polar[0, :, pidx]
        #[1, 100]
        
        m1 = torch.tensor(b3, device=device, dtype=torch.float32) 
        tp = torch.tensor(tp, device=device, dtype=torch.long)
        oc, op, ot, ow, mu, logvar, ploc = model(wave)
        #print("op",op.shape) #[128, 2, 10240]
        #print(b1.shape)  #(128, 10240)
        #print(b1)
        #print(oc.shape, op.shape, ot.shape, d.shape, d1.shape, m1.shape, tp.shape)
        loss, loss1, loss2, loss3, loss4, loss5 = lossfn(
            oc, d, op, d1, m1, ot, tp, ow, wave, mu, logvar, eloc, ploc)  #计算模型输出与标签之间的损失
        loss.backward()
        if loss.isnan():
            print("NAN error")
            optim.zero_grad()
            break  
        optim.step() 
        optim.zero_grad()
        ls = loss.detach().cpu().numpy()
        ls1 = loss1.detach().cpu().numpy()
        ls2 = loss2.detach().cpu().numpy()
        ls3 = loss3.detach().cpu().numpy()
        ls4 = loss4.detach().cpu().numpy()
        ls5 = loss5.detach().cpu().numpy()
        ed = time.perf_counter() #记录当前时间，用于计算每个训练步骤的执行时间
        outloss.write(f"{step},{ed - st},{ls},{ls1},{ls2},{ls3},{ls4},{ls5}\n") #训练步骤、执行时间和损失值写入损失记录文件
        outloss.flush()
        acc_time += ed - st
        
            
        if step % 100 == 0: #每训练100步执行
            torch.save(model.state_dict(), model_name) #保存当前模型的权重
            #d1 = d1.detach().cpu().numpy().reshape(-1)
            #op [128,2,10240] op[0][1]为模型判定向下的概率分布
            #p1 = np.argmax(op.detach().cpu().numpy(), axis=1).reshape(-1) 
            #print("d1",len(d1))
            #print("p1",len(p1))
            #m1 = m1.detach().cpu().numpy().reshape(-1)
            #dd1 = d1[m1==1] 
            #pp1 = p1[m1==1] 
            #if len(dd1)==0:
            #    dd1 = np.zeros([3])
            #    pp1 = np.ones([3])
            print(f"{acc_time:6.1f}, {step:8}, Loss:{ls:.6f},\nPhase{ls1:.6f},Polar{ls2:.6f},Type{ls3:.6f},Wave{ls4:.6f},Loc{ls5:.6f}")
            gs = gridspec.GridSpec(6, 1,hspace=0.3) 
            fig = plt.figure(1, figsize=(16, 16), dpi=100) 

            p = oc.detach().cpu().numpy()[0]
            
            d = a3[0]
            ow = ow.detach().cpu().numpy()  
            op = op.softmax(dim=1).detach().cpu().numpy()
            m1 = m1.detach().cpu().numpy() 
            d1 = d1.detach().cpu().numpy() 
            ax = fig.add_subplot(gs[5, 0])
            #print(op.shape, m1.shape, d1.shape)
            ax.plot(op[0, 0, :], c="r", alpha=0.5, label="Pred.")
            ax.plot(m1[0, :],    c="g", alpha=0.5, label="MASK")
            ax.plot(d1[0, :],    c="b", alpha=0.5, label="LABEL")
            ax.legend()
            ax.set_ylim(-0.2, 1.2)
            for i in range(5):
                ax = fig.add_subplot(gs[i, 0])
                ax.set_ylim(-1.2, 1.2)
                if i == 0:ax.set_title("Noise", ha="left", va="bottom", x=0.00, y=1.01)
                if i == 1:
                    ax.set_title("Pg", ha="left", va="bottom", x=0.00, y=1.01)       
                if i == 2:ax.set_title("Sg", ha="left", va="bottom", x=0.00, y=1.01)
                if i == 3:ax.set_title("Pn", ha="left", va="bottom", x=0.00, y=1.01)
                if i == 4:ax.set_title("Sn", ha="left", va="bottom", x=0.00, y=1.01)
                #max_index = np.argmax(d[:, 1])
                
                ax.plot(d[:, i], alpha=0.5, c="b") # 绘制震相标签，用蓝色表示
                ax.plot(p[i, :], alpha=0.9, c="r") #绘制预测，用红色表示
                #if i == 5:
                #    ax.plot(polarweigh, alpha=0.5, c="g") 
                #    ax.plot(polartype, alpha=0.5, c="b")
                #if i == 6:ax.plot(modelpolar[0, :], alpha=0.5, c="g")
                #if i == 7:ax.plot(polarweigh[1, :], alpha=0.5, c="g")
                w = a1[0, :, i%3]
                #w /= (np.max(w)+1e-6)
                w2 = ow[0, i%3, :]
                ax.plot(w, alpha=0.3, c="b")
                ax.plot(w2, alpha=0.3, c="r")
                #ax.set_xlim(max_index - 1000, max_index + 1000)
            #ax = fig.add_subplot(gs[, 0])
            plt.savefig(f"logdir/trainfig/Large/largev9b.{version}.png") 
            plt.cla() 
            plt.clf()
            acc_time = 0 
        #if ls < 22000:break
    print("done!")
#nohup python china.large9o.steps.py > logdir/large9o.log 2>&1 &
#2289209
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拾取连续波形")          
    parser.add_argument('-d', '--dist', default=200, type=int, help="输入连续波形")       
    parser.add_argument('-o', '--output', default="result/t1", help="输出文件名")      
    parser.add_argument('-m', '--model', default="lppn.model", help="模型文件lppnmodel")                                                            
    args = parser.parse_args()      
    main(args)




