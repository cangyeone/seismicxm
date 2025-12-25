
import time
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import torch 
from utils.forplt import LargeDataTest2PNW
from models.EQLarge9 import EQLargeCNN as Model, Loss 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150

from scipy.signal import find_peaks 



def main(args):
    partion = "test"
    steps = "200"
    model_name = f"testckpt/transfer.pnw.balanced.{steps}.pt" #保存和加载神经网络模型权重的文件路径
    data_tool = LargeDataTest2PNW(partion)
    device = torch.device("mps")
    model = Model()
    model.load_state_dict(torch.load(model_name, map_location=device))
    dtype = torch.float32
    model.to(device)
    model.eval()
    phase_name_file = "large/event.type"
    with open(phase_name_file, "r") as f:
        phase_dict = eval(f.read())
        id2p = {} 
        for k, v in phase_dict.items():
            id2p[v] = k 
    st = time.perf_counter()
    ofile = open(f"odata/v9.transfer.pnw.balanced.{partion}.{steps}.txt", "w")
    for i in range(500):
        x, d, c = data_tool.batch_data(10)
        x = torch.tensor(x, dtype=torch.float32, device=device) 
        d = torch.tensor(d, dtype=torch.long, device=device)
        print(x.shape)
        phase, polar, event_type, wave, mu, logvar, _ = model(x.permute(0, 2, 1))
        
        p = event_type.argmax(dim=1) 
        d = d.squeeze().long() 
        p = p.cpu().detach().numpy() 
        d = d.cpu().detach().numpy()
        for a1, a2, a3 in zip(p, d, c):
            ofile.write(f"{a1},{a2},{a3}\n")
        print(i, a1, a2, a3)
    acc = np.mean(p==d)
    print(f"精度：{acc:.3f}")
    print("混淆矩阵")
    print("  ", ",".join([id2p[i] for i in range(3)]))
    for i in range(3):
        nums = []
        for k in range(3):
            num = np.sum((d==i)*(p==k))
            nums.append(f"{num:02}")
        print(f"{id2p[i]},{','.join(nums)}")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拾取连续波形")          
    parser.add_argument('-d', '--dist', default=200, type=int, help="输入连续波形")       
    parser.add_argument('-o', '--output', default="result/t1", help="输出文件名")      
    parser.add_argument('-m', '--model', default="lppn.model", help="模型文件lppnmodel")                                                            
    args = parser.parse_args()      
    main(args)




