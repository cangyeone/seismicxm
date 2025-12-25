from cmath import polar
import time
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import torch 
from utils.forplt import LargeData2
from models.EQLarge9a import EQLargeCNN as Model, Loss 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150

import torch.nn as nn
def main(args):
    version = "large.middle"#7.72
    model_name = f"ckpt/middle.pt" #保存和加载神经网络模型权重的文件路径
    data_tool = LargeData2(istest=False)
    #data_tool = DataPnSnWithPolarType(file_name="h5data", stride=1, n_length=20480, padlen=4096)
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
            if "decoder_event_type" in key: # 仅有最后一层有out
                var.requires_grad = True
            else:
                var.requires_grad = False  
    model.to(device)
    model.train()
    lossfn = Loss() 
    lossfn.to(device)
    acc_time = 0 #记录训练的累计时间
    outloss = open(f"logdir/largev9.{version}.txt", "a") #记录训练过程中的loss
    if isfixed:
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 1e-5, weight_decay=0e-4)
    else:
        optim = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-3)
    
    phase_name_file = "large/phase.type"
    with open(phase_name_file, "r") as f:
        phase_dict = eval(f.read())
        id2p = {0:"Noise"} 
        for k, v in phase_dict.items():
            id2p[v+1] = k 
    lossfn = nn.CrossEntropyLoss()
    for step in range(5000):
        st = time.perf_counter()
        x, d = data_tool.batch_data(256)
        
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
            print(f"{step},{ed - st},{ls},{np.mean(p==d)}\n")
            outloss.write(f"{step},{ed - st},{ls}\n")
            outloss.flush()
            acc_time += ed - st
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


