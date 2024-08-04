import torch 
from prime.middle import PRIMEDP 

model = PRIMEDP() 
model.load_state_dict(torch.load("ckpt/primedp.middle.pt"))

x = torch.randn([32, 3, 10240])# N, C, T format. 
phase, polar, event_type, wave, hidden = model(x) 
