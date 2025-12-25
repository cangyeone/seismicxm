from cmath import polar
import time
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import torch 
from utils.forplt import LargeData2
from models.EQLarge9 import EQLargeCNN as Model2, Loss 
plt.switch_backend('agg')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['figure.dpi'] = 150

import torch.nn as nn

class Model(Model2):
    def forward(self, x):
        N, C, T = x.shape 
        x -= torch.mean(x, dim=2, keepdim=True)
        maxx, maxidx = torch.max(torch.abs(x), dim=2, keepdim=True) 
        maxx, maxidx = torch.max(maxx, dim=1, keepdim=True) 
        x /= (maxx + 1e-6)  
        #x = x[:, :, :3]
        xemb_wave = self.embedding_wave(x)
        xemb_wave = self.emb_pos(xemb_wave)
        #print(xemb_wave.shape)
        #xemb_wave = xemb_wave.permute(0, 2, 1)
        pos_id = torch.arange(6).long().unsqueeze(0).repeat(N, 1).to(x.device)
        emb_pos = self.embedding_info(pos_id)
        xemb_pad = torch.cat([xemb_wave, emb_pos], dim=1)

        hidden = self.transformer(xemb_pad) 
        hidden = hidden.permute(0, 2, 1)
        #print(hidden.shape)
        hidden_wave = hidden[:, :, 6:]
        hidden_einfo = hidden[:, :, :6] 
        phase = self.decoder_phase(hidden_wave)#.sigmoid()
        phase = phase.softmax(dim=1)

        # 变分自编码器求
        #mu     = hidden_wave[:, :self.n_feature//2, :]
        #logvar = hidden_wave[:, self.n_feature//2:, :]
        #std = torch.exp(0.5 * logvar) 
        #eps = torch.randn_like(std) 
        #s = eps * std + mu 
        wave = self.decoder_wave(hidden_wave)
        wave = torch.tanh(wave)
        
        polar = self.decoder_polar(hidden_wave)
        event_type = self.decoder_event_type(hidden_einfo[:, :, 0])
        event_pos_x = self.decoder_position_x(hidden_einfo[:, :, 1]).sigmoid() * 2000 
        #event_pos_z = self.decoder_position_z(hidden_einfo[:, :, 2])
        #event_type = self.decoder_event_type(hidden_einfo[:, :, 3])
        #event_mag = self.decoder_event_mag(hidden_einfo[:, :, 4]).sigmoid()*12 - 2
        #event_delta = self.decoder_event_delta(hidden_einfo[:, :, 5]).sigmoid() * 20
        return phase.softmax(dim=1), polar.softmax(dim=1), event_type.softmax(dim=1), wave, event_pos_x

def main(args):
    version = "04b"
    model_name = "ckpt_large/seismicxm.middle.classification.pt"  # 保存和加载神经网络模型权重的文件路径
    model = Model()
    model.load_state_dict(torch.load(model_name, map_location="cpu"))
    torch.jit.save(torch.jit.script(model), f"picker/largev9.jit")  
main([])
# Input of the jit: [BatchSize, 3, 10240]
# Output: Phase[BS, 5(Noise, Pg, Sg, Pn, Sn), 10240], Polar[BS, 2, 10240], EventType[BS, 8], Wave, EventPos
