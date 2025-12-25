import torch 
import torch.nn as nn 
import torch.nn.functional as F

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.dropout = nn.Dropout(dropout)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class ConvBNReLU(nn.Module):
    def __init__(self, nin, nout, ks, stride) -> None:
        super().__init__() 
        pad = (ks-1)//2
        self.layers = nn.Sequential(
            nn.Conv1d(nin, nout, ks, stride=stride, padding=pad), 
            nn.BatchNorm1d(nout), 
            nn.GELU(), 
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 

class ConvBNTReLU(nn.Module):
    def __init__(self, nin, nout, ks, stride=2) -> None:
        super().__init__() 
        pad = (ks-1)//2
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(nin, nout, ks, stride, padding=(ks-1)//2, output_padding=stride-1), 
            nn.BatchNorm1d(nout), 
            nn.GELU(), 
        )
    def forward(self, x):
        x = self.layers(x) 
        return x 

class Encoder(nn.Module):
    def __init__(self, n_in=35, n_out=768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNReLU(n_in, 32, 5, 2), 
            ConvBNReLU(32, 64, 5, 2), 
            ConvBNReLU(64, 128, 5, 2), 
            #ConvBNReLU(128, 128, 5, 2), 
            ConvBNReLU(128, 256, 5, 2), 
            #ConvBNReLU(256, 256, 5, 2), 
            ConvBNReLU(256, 512, 5, 2), 
            #ConvBNReLU(512, 512, 5, 2), 
            ConvBNReLU(512, n_out, 5, 2),    
        )
    def forward(self, x):
        x = self.layers(x)
        return x 
class Decoder(nn.Module):
    def __init__(self, n_in=768, n_out=37) -> None:
        super().__init__() 
        self.layers = nn.Sequential(
            ConvBNTReLU(n_in, 512, 5, 2),#2 
            #ConvBNTReLU(512, 512, 5, 2), #4
            ConvBNTReLU(512, 256, 5, 2), #8
            #ConvBNTReLU(256, 256, 5, 2), #16
            ConvBNTReLU(256, 128, 5, 2), #32
            #ConvBNTReLU(128, 128, 5, 2), #64
            ConvBNTReLU(128, 64, 5, 2),  #128
            ConvBNTReLU(64, 64, 5, 2),   
            ConvBNTReLU(64, 32, 5, 2), 
            nn.Conv1d(32, n_out, 5, 1, padding=2), 
            #nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layers(x)
        return x 

class DecoderPhase(nn.Module):
    def __init__(self, n_in=768, n_out=37) -> None:
        super().__init__() 
        self.layers = nn.Sequential(
            ConvBNTReLU(n_in, 512, 5, 2),#2 
            #ConvBNTReLU(512, 512, 5, 2), #4
            ConvBNTReLU(512, 256, 5, 2), #8
            #ConvBNTReLU(256, 256, 5, 2), #16
            ConvBNTReLU(256, 128, 5, 2), #32
            #ConvBNTReLU(128, 128, 5, 2), #64
            ConvBNTReLU(128, 64, 5, 2),  #
            ConvBNTReLU(64, 64, 5, 2), 
            ConvBNTReLU(64, 32, 5, 2), 
            nn.Conv1d(32, n_out, 5, 1, padding=2), 
            #nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layers(x)
        return x 
class BRNNIN(nn.Module):
    def __init__(self, nin=64, nout=16) -> None:
        super().__init__() 
        self.rnn = nn.LSTM(nin, nout, 1, bidirectional=True) 
        self.cnn = nn.Conv1d(nout*2, nout, 1)
        self.layernorm = nn.LayerNorm(nout*2)
        self.nout = nout 
    def forward(self, x):
        B, C, T = x.shape 
        #x = x.permute(0, 2, 1)
        x = x.permute(2, 0, 1)
        h0 = torch.zeros([2, B, self.nout], dtype=x.dtype, device=x.device)
        c0 = torch.zeros([2, B, self.nout], dtype=x.dtype, device=x.device)
        h, (h0, c0) = self.rnn(x, (h0, c0)) 
        h = self.layernorm(h)
        # T, B, C, 
        h = h.permute(1, 2, 0)
        #h = h.permute(0, 2, 1)
        y = self.cnn(h) 
        y = y.permute(0, 2, 1)
        return y 
import torch.nn.functional as F 
class EQLargeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        n_stride = 512
        n_feature = 256 
        self.n_feature = n_feature 
        self.embedding_wave = Encoder(n_in=3, n_out=n_feature)#nn.Conv1d(35, n_feature, n_stride, n_stride)
        self.emb_pos = BRNNIN(n_feature, n_feature)
        self.embedding_info = nn.Embedding(num_embeddings=32, embedding_dim=n_feature)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_feature, dim_feedforward=n_feature, activation=F.gelu, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        #self.transformer = BRNNIN(n_feature, n_feature)
        self.decoder_phase = DecoderPhase(n_feature, 5)
        self.decoder_wave = Decoder(n_feature, 3)
        self.decoder_polar = Decoder(n_feature, 2)
        self.decoder_event_mag = nn.Linear(n_feature, 1) 
        self.decoder_position_x = nn.Linear(n_feature, 4)
        self.decoder_position_z = nn.Linear(n_feature, 1)
        self.decoder_event_type = nn.Linear(n_feature, 8)
        self.decoder_event_delta = nn.Linear(n_feature, 1)
        
    def forward(self, x, task_id=None):
        N, C, T = x.shape 
        #x = x[:, :, :3]
        xemb_wave = self.embedding_wave(x)
        xemb_wave = self.emb_pos(xemb_wave)
        #print(xemb_wave.shape)
        #xemb_wave = xemb_wave.permute(0, 2, 1)
        if task_id is None:
            pos_id = torch.arange(6).long().unsqueeze(0).repeat(N, 1).to(x.device)
            B, F = pos_id.shape 
        else:
            pos_id = task_id.long().unsqueeze(0).repeat(N, 1).to(x.device)
            B, F = pos_id.shape 
        emb_pos = self.embedding_info(pos_id)
        xemb_pad = torch.cat([xemb_wave, emb_pos], dim=1)

        hidden = self.transformer(xemb_pad) 
        hidden = hidden.permute(0, 2, 1)
        #print(hidden.shape)
        hidden_wave = hidden[:, :, F:]
        hidden_einfo = hidden[:, :, :F] 
        if task_id is None:
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
            return phase, polar, event_type, wave, hidden 
        else:
            return hidden_einfo, hidden_wave 


class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.l1 = nn.L1Loss()
    def forward(self, x, d, x1, d1, m1, x2, d2, ow, wave, mu, logvar):
        m = d[:, 1:, :].clone()
        # 1 2 4 8
        m[:, 0, :] *= 1
        m[:, 1, :] *= 1
        m[:, 2, :] *= 2
        m[:, 3, :] *= 4
        m = torch.sum(m, dim=1)
        B, C, T = x.shape 
        loss = (d * torch.log(x+1e-9)).sum(dim=1) * (m * 1 + 1) # 加权加大
        loss0 = - (loss).sum() / B 
        #loss0 = - (d * torch.log(x+1e-9)).sum() / B#震相
        loss1 = self.ce(x1, d1) * m1 
        loss1 = loss1.sum() 
        #loss1 = loss1.sum()  * 80
    
        loss2 = self.ce(x2, d2) 
        loss2 = loss2.sum() * 10  * B/256
        #loss2 = loss2.sum() * 10240
        loss31 = (torch.square(ow - wave)).sum() / B /100
        loss32 = torch.sum(
                -1-logvar + mu ** 2 + logvar.exp(), dim = 1
            ).sum() / B /100
        loss3 = loss31 +  loss32 * 0.01 

        loss = loss0 + loss1  + loss2 + loss3
        #print(loss0,loss1,loss2)
        return loss, loss0, loss1, loss3 
        
if __name__ == "__main__":
    model = EQLargeCNN()
    x = torch.ones([32, 3, 10240]) 
    y1, y2, y3  = model(x)
    print(y1.shape, y2.shape, y3.shape)