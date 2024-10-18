import torch 
import torch.nn as nn 
from obspy import read
from glob import glob
import os 
import numpy as np
from models.EQLarge9 import EQLargeCNN as Model, Loss 
class Picker(Model):
    def __init__(self):
        super().__init__()
        self.n_stride = 1 
    def forward(self, x):
        device = x.device
        with torch.no_grad():
            #print("数据维度", x.shape)
            T, C = x.shape 
            seqlen = 10240 
            batchstride = 10240 - 4096
            batchlen = torch.ceil(torch.tensor(T / batchstride).to(device))
            idx = torch.arange(0, seqlen, 1, device=device).unsqueeze(0) + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride 
            idx = idx.clamp(min=0, max=T-1).long()
            x = x.to(device)
            wave = x[idx, :] 
            wave = wave.permute(0, 2, 1)
            wave -= torch.mean(wave, dim=2, keepdim=True)
            #max = torch.std(wave, dim=2, keepdim=True)
            max, maxidx = torch.max(torch.abs(wave), dim=2, keepdim=True) 
            wave /= (max + 1e-6)  
            #print(wave.shape)
            xemb_wave = self.embedding_wave(wave)
            xemb_wave = self.emb_pos(xemb_wave)
            #print(xemb_wave.shape)
            #xemb_wave = xemb_wave.permute(0, 2, 1)
            B, C, T = xemb_wave.shape
            pos_id = torch.arange(6).long().unsqueeze(0).repeat(B, 1).to(x.device)
            emb_pos = self.embedding_info(pos_id)
            xemb_pad = torch.cat([xemb_wave, emb_pos], dim=1)
            #xemb_pad = xemb_pad.permute(0, 2, 1)
            hidden = self.transformer(xemb_pad) 
            hidden = hidden.permute(0, 2, 1)
            #print(hidden.shape)
            hidden_wave = hidden[:, :, 6:]
            hidden_einfo = hidden[:, :, :6] 
            y1 = self.decoder_phase(hidden_wave)#.sigmoid()
            y2 = self.decoder_polar(hidden_wave)
            y3 = self.decoder_event_type(hidden_einfo[:, :, 0])
            oc = y1.softmax(dim=1)
            op = y2.softmax(dim=1)          
            ot = y3.softmax(dim=1)        
            B, C, T = oc.shape 
            tgrid = torch.arange(0, T, 1, device=device).unsqueeze(0) * self.n_stride + torch.arange(0, batchlen, 1, device=device).unsqueeze(1) * batchstride
            oc = oc.permute(0, 2, 1).reshape(-1, C) 
            op = op.permute(0, 2, 1).reshape(-1, 2)
            ot = tgrid.squeeze()
            ot = ot.reshape(-1)
            output = []
            outpol = []
            #print("NN处理完成", oc.shape, ot.shape)
            # 接近非极大值抑制（NMS） 
            # .......P........S...... 
            #print("SHAPE", op.shape, oc.shape, ot.shape)
            probs = [0.2, 0.2, 0.2, 0.2, 0.2]#不同类型置信度，Pg，Sg，Pn，Sn
            for itr in range(4):
                pc = oc[:, itr+1] 
                time_sel = torch.masked_select(ot, pc>probs[itr])
                score = torch.masked_select(pc, pc>probs[itr])
                pol = torch.masked_select(op[:, 0], pc>probs[itr])
                
                _, order = score.sort(0, descending=True)    # 降序排列
                ntime = time_sel[order] 
                nprob = score[order]
                npolor = pol[order]
                #print(batchstride, ntime, nprob)
                select = -torch.ones_like(order)
                selidx = torch.arange(0, order.numel(), 1, dtype=torch.long, device=device) 
                count = 0
                while True:
                    if nprob.numel()<1:
                        break 
                    ref = ntime[0]
                    idx = selidx[0]
                    select[idx] = 1 
                    count += 1 
                    selidx = torch.masked_select(selidx, torch.abs(ref-ntime)>300)
                    nprob = torch.masked_select(nprob, torch.abs(ref-ntime)>300)
                    ntime = torch.masked_select(ntime, torch.abs(ref-ntime)>300)
                    #if itr == 0:
                p_time = torch.masked_select(time_sel[order], select>0.0)
                p_prob = torch.masked_select(score[order], select>0.0)
                p_type = torch.ones_like(p_time) * itr 
                y = torch.stack([p_type, p_time, p_prob], dim=1)
                output.append(y) 
                #if itr == 0:
                pols = torch.masked_select(pol[order], select>0.0)
                outpol.append(pols)
            y1 = torch.cat(output, dim=0)
            y2 = torch.cat(tensors=outpol, dim=0)
        return y1 
version = "05.step2"
model_name = f"ckpt_large/largev9.{version}.pt" #保存和加载神经网络模型权重的文件路径, ckpt_large/largev9.05.step1.pt
    #ckpt_large/largev8.02.step3.pt
jit_name = f"pickers/largev9.{version}.jit"
model = Picker() 
model.load_state_dict(torch.load(model_name, map_location="cpu"))
model.eval()
torch.jit.save(torch.jit.script(model), jit_name)
x = torch.randn([300000, 3])
y = model(x)
#print(y)
#phase_dict = {'Sg': 1, 'Pg': 0, 'Pn': 2, 'Sn': 3, 'SKKS': 4, 'SKS': 5, 'SS': 6, 'PKS': 7, 'PP': 8, 'pPKP': 9, 'sPKP': 10, 'ePP': 11, 'eSS': 12, 'PKP': 13, 'f': 14, 'eP': 15, 'P': 16, 'ePn': 17, 'ePKP': 18, 'ePg': 19, 'S': 20, 'pP': 21, 'sP': 22, 'sS': 23, 'eS': 24, 'esP': 25, 'PcP': 26, 'ScP': 27, 'ScS': 28, 'PcS': 29, 'ePcP': 30, 'epP': 31, 'PmP': 32, 'Pb': 33, 'SmS': 34, 'Sb': 35, "Other":36}
#event_dict = {'eq': 0, 'ep': 1, 'ss': 2, 'sp': 3, 'ot': 4, 'se': 5, 've': 6}
