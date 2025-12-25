
import time
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import torch 
from utils.forplt import LargeDataTest2, LargeDataTest2PNW 
from models.EQLarge9 import EQLargeCNN as Model, Loss 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150

from scipy.signal import find_peaks 

import tqdm 

def main(args):
    partion = "test"
    steps = "200"
    model_name1 = f"testckpt/orignal.200.pt" #保存和加载神经网络模型权重的文件路径
    model_name2 = f"testckpt/transfer.200.pt"
    model_name3 = f"testckpt/orignal.pnw.200.pt"
    model_name4 = f"testckpt/transfer.pnw.200.pt"
    model_name5 = f"testckpt/orignal.pnw.balanced.200.pt"
    model_name6 = f"testckpt/transfer.pnw.balanced.200.pt"
    
    model_names = [model_name1, model_name2, model_name3, model_name4, model_name5, model_name6]
    file_path = [tstr.replace("testckpt", "odata/cross")+".data1.txt" for tstr in model_names]
    ofiles1 = [open(path, "w") for path in file_path]
    file_path = [tstr.replace("testckpt", "odata/cross")+".data2.txt" for tstr in model_names]
    ofiles2 = [open(path, "w") for path in file_path]
    data_tool1 = LargeDataTest2(partion)
    data_tool2 = LargeDataTest2PNW(partion)
    device = torch.device("mps")
    
    models = [Model() for _ in range(6)]
    
    #model = Model()
    for idx, mname in enumerate(model_names):
        models[idx].load_state_dict(torch.load(mname, map_location="cpu"))
        models[idx].to(device)
        models[idx].eval()
    
    phase_name_file = "large/event.type"
    with open(phase_name_file, "r") as f:
        phase_dict = eval(f.read())
        id2p = {} 
        for k, v in phase_dict.items():
            id2p[v] = k 
    st = time.perf_counter()
    ndict = {0:1, 1:0, 2:2}
    for i in tqdm.tqdm(range(176)):
        x, d, c = data_tool1.batch_data(5)
        x = torch.tensor(x, dtype=torch.float32, device=device) 
        d = torch.tensor(d, dtype=torch.long, device=device)
        d = d.squeeze().long() 
        d = d.cpu().detach().numpy()
        for idx, (ofile, model) in enumerate(zip(ofiles1, models)):
            phase, polar, event_type, wave, mu, logvar, _ = model(x.permute(0, 2, 1))
            
            p = event_type.argmax(dim=1) 
            p = p.cpu().detach().numpy() 
            for a1, a2, a3 in zip(p, d, c):
                if idx in [0, 1]:
                    pass 
                else:
                    a1 = ndict[int(a1)]
                ofile.write(f"{a1},{a2},{a3}\n")
    
    for i in tqdm.tqdm(range(100)):
        x, d, c = data_tool2.batch_data(50)
        x = torch.tensor(x, dtype=torch.float32, device=device) 
        d = torch.tensor(d, dtype=torch.long, device=device)
        d = d.squeeze().long() 
        d = d.cpu().detach().numpy()
        for idx, (ofile, model) in enumerate(zip(ofiles2, models)):
            phase, polar, event_type, wave, mu, logvar, _ = model(x.permute(0, 2, 1))
            
            p = event_type.argmax(dim=1) 
            p = p.cpu().detach().numpy() 
            
            for a1, a2, a3 in zip(p, d, c):
                if idx in [1, 2]:
                    #a1 = ndict[int(a1)]
                    a2 = ndict[int(a2)] 
                else:
                    a1 = ndict[int(a1)]
                    a2 = ndict[int(a2)]
                ofile.write(f"{a1},{a2},{a3}\n")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拾取连续波形")          
    parser.add_argument('-d', '--dist', default=200, type=int, help="输入连续波形")       
    parser.add_argument('-o', '--output', default="result/t1", help="输出文件名")      
    parser.add_argument('-m', '--model', default="lppn.model", help="模型文件lppnmodel")                                                            
    args = parser.parse_args()      
    main(args)




