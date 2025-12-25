
from datetime import date
import time
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import torch 
#from models.LAEGE import BRNN as Model, Loss 
#from models.EQLarge2 import EQLargeCNN as Model, Loss 
#from models.EQLarge6 import EQLargeCNN as Model, Loss 
from models.EQLarge9 import EQLargeCNN as Model
import scipy.signal as signal  
#from utils.large7 import DataLarge9step2Test
import os
# compare_seisbench_pickers.py
import obspy
import numpy as np
import numpy as np
import obspy
from obspy import UTCDateTime
import seisbench.models as sbm
from typing import Dict, List, Tuple

def find_phase2(pred, delta=1.0, height=0.80,pgh=0.1,sgh=0.1,pnh=0.1,snh=0.1, dist=1):
    pred = pred.cpu().numpy() #预测数据
    shape = np.shape(pred)  
    #print(shape)  (1000, 5, 10240)
    all_phase = []
    #phase_name = {0:"N", 1:"P", 2:"S"}
    for itr in range(shape[0]):
        #print(itr) 0 -- batch_size
        phase = []
        for itr_c in [0, 1, 2, 3]:
            #分别寻找对应的Pg，Sg，Pn，Sn的峰值
            p = pred[itr, itr_c+1, :]

            #print(p) 
            #p = signal.convolve(p, np.ones([10])/10., mode="same")
            #print(h)
            if itr_c == 0:
                peaks, _ = signal.find_peaks(p, height=pgh, distance=dist) 
            if itr_c == 1:
                peaks, _ = signal.find_peaks(p, height=sgh, distance=dist) 
            if itr_c == 2:
                peaks, _ = signal.find_peaks(p, height=pnh, distance=dist) 
            if itr_c == 3:
                peaks, _ = signal.find_peaks(p, height=snh, distance=dist)                 
            for itr_p in peaks:
                phase.append(
                    [
                        itr_c+1, #phase_name[itr_c], 
                        itr_p, 
                        pred[itr, itr_c+1, itr_p], 
                        itr_p
                    ]
                    )
        all_phase.append(phase)
    return all_phase 


# ============== 3) 统一的“模型运行器” ==============
def run_picker(model, st_in: obspy.Stream, batch_size=32):
    """
    - 自动 Z/N/E 排序与按模型采样率重采样
    - 调用 classify()，从结果对象的 .picks 里取出拾取
    - 返回：[(phase, timeUTC, prob, sample_index_at_input_fs), ...]
    """
    # 组件与顺序
    req = getattr(model, "required_components", ("Z", "N", "E"))
    comp_map = {"Z": None, "N": None, "E": None}
    for tr in st_in:
        ch = tr.stats.channel[-1].upper()
        if ch in comp_map and comp_map[ch] is None:
            comp_map[ch] = tr
    assert all(comp_map[c] is not None for c in "ZNE"), "输入必须包含 Z/N/E 三个分量（通道末位为 Z/N/E）"

    st = obspy.Stream([comp_map[c] for c in req])

    # 重采样到模型采样率
    model_fs = float(getattr(model, "sampling_rate", st[0].stats.sampling_rate))
    if abs(model_fs - st[0].stats.sampling_rate) > 1e-6:
        st = st.copy()
        st.resample(model_fs)

    out = model.classify(st, batch_size=batch_size)  # 结果对象
    picks = getattr(out, "picks", [])                # PickList

    fs_in = st_in[0].stats.sampling_rate
    t0 = st_in[0].stats.starttime
    results = []
    for p in picks:
        # Pick: start_time / end_time / peak_time / phase / peak_value
        t_abs = getattr(p, "peak_time", None) or p.start_time
        prob  = getattr(p, "peak_value", float("nan"))
        idx   = int(round((t_abs - t0) * fs_in))
        results.append((p.phase.upper() if p.phase else "?", t_abs, float(prob), idx))
    return results

def load_model(name, weight, extra):
    Cls = getattr(sbm, name)
    print(Cls.list_pretrained())
    return Cls.from_pretrained(weight, **extra)

# ============== 2) 构造成 ObsPy Stream（确保通道名是 Z/N/E） ==============
def build_stream(xN, xE, xZ, fs: float, t0: UTCDateTime) -> obspy.Stream:
    st = obspy.Stream(traces=[
        obspy.Trace(data=xZ.copy(), header={"channel": "BHZ", "starttime": t0, "sampling_rate": fs}),
        obspy.Trace(data=xN.copy(), header={"channel": "BHN", "starttime": t0, "sampling_rate": fs}),
        obspy.Trace(data=xE.copy(), header={"channel": "BHE", "starttime": t0, "sampling_rate": fs}),
    ])
    # 轻量预处理：去均值/去趋势；如需可添加带通，例如 1-20 Hz（视区域频带而定）
    st.detrend("demean"); st.detrend("linear")
    # 可选：幅度标准化（对极端值更鲁棒）
    for tr in st:
        s = float(np.std(tr.data)) or 1.0
        tr.data = np.clip(tr.data / s, -20, 20)
    return st
import tqdm 
def main(args):

    pgh = 0.1
    sgh = 0.1
    pnh = 0.1
    snh = 0.1
    
    basedir = "data/testv9"
    names = os.listdir(basedir)
    names = names[:2500]
    device = torch.device("mps")
    #names = range(1000)
    MODEL_SPECS = [
        #("PhaseNet-geofon",  "PhaseNet",      "original",   {}),                 # 第4列预留额外kwargs
        ("GPD-original-v2",     "GPD",           "original", {}),
        #("EQTransformer", "EQTransformer", "stead", {}),   # 关键：version=3
    ]
    ofiles = [open(f"odata/v9.{name}.phase.txt", "w") for name, _, _, _ in MODEL_SPECS]
    for idx, (disp_name, cls_name, weight, extra) in enumerate(MODEL_SPECS):
        print(f"\n==> Loading {disp_name}")
        model = load_model(cls_name, weight, extra)
        model = model.to(device)
        f = ofiles[idx]
        for step in tqdm.tqdm(names):
            #a1：使用的数据  a2：标签数据
            #a1, a2, a3, a4, a5 = data_tool.batch_data(batch_size=100)
            #a1, a2, a3, a4, a5 = data_tool.batch_data(100) 
            fz = np.load(os.path.join(basedir, step))
            a1, a2, a3, a4, a5 = fz["a1"], fz["a2"], fz["a3"], fz["a4"], fz["a5"]
            #print(f"Processing:{step}")
            #print(phase)
            for idx in range(len(a2)):
                wave = a1[idx] 
                st_in = build_stream(wave[:, 0], wave[:, 1], wave[:, 2], fs=100.0, t0=UTCDateTime(2020,1,1))
                results = run_picker(model, st_in, batch_size=64)
                #print(a2[idx])
                pt, st, pn, sn = a2[idx] 
                p, c = a3[idx]
                snr = a4[idx] 
                dist = a5[idx]
                f.write(f"#phase,{pt},{st},{pn},{sn},{p},{c},{snr[0]},{snr[1]},{snr[2]},{snr[3]},{dist}\n") 
                #print(f"#phase,{pt},{st},{pn},{sn},{snr}\n")
                for p in results:
                    if p[0] in ("P", "Pg"):
                        ptype = 0 
                    elif p[0] in ("S", "Sg"):
                        ptype = 1
                    else:
                        continue 
                    #if int(p[0]) == 1:
                    f.write(f"{ptype},{p[3]},{p[2]},0.0\n") 
                f.flush()
            time2 = time.perf_counter()
            #acctime[num, step%50] = time3 - time1
            #ttt = np.mean(acctime, axis=1)*1000
            #tlist = [f"{n.replace('.pt', '').replace('diting.', '')}:{i:.3f}" for n, i in zip(mnames, ttt)]
            #tstrs = ",".join(tlist)
            #print(f"Finished:{step}")
        f.close()



import argparse
if __name__ == "__main__":
    main([])


