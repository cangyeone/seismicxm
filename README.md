### PRIME-DP: Pre-trained Integrated Model for Earthquake Data Processing
Our model is based on Transformer. And is mainly used to get seismic wave repersation. 
![structure of PRIME-DP](fig/structure.en.png)

Here are five models:

|Name|Number of parameters|path|status|
|:-:|:-:|:-:|:-:|
|RNN model|77M|ckpt/primedp.rnn.pt|released|
|Picker model|0.5M|ckpt/primedp.picker.pt|released|
|Tinny model|8.6M|ckpt/primedp.tinny.pt|released|
|Middle model|51M|ckpt/primedp.middle.pt|released|
|Event classification model based on middle|51M|ckpt/primedp.middle.classification.pt|released|
|Large model|1.3B|ckpt/primedp.large.pt|training|

### 1. Usage
```Python 
from prime.middle import PRIMEDP 
import torch 

model = PRIMEDP() 
# load pretrained model 
model.load_state_dict(torch.load("ckpt/primedp.middle.pt"))
# model input 
x = torch.randn([32, 3, 10240])# N, C, T顺序
# phase, polarity, event type, waveform, featrue vector 
phase, polar, event_type, wave, hidden = model(x) 
# can be used for other tasks. 
```

### 2. Transfer learning 
Take event classification as example: 
```Python 
# define the trainig parameters 
for key, var in model.named_parameters():
    if var.dtype != torch.float32:continue # BN layers
    if "decoder_event_type" in key: # classification decoder 
        var.requires_grad = True
    else:
        var.requires_grad = False  
# define the optimizer 
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 1e-3, weight_decay=1e-1)
```

the weights "ckpt\primedp.middel.classification.pt" are trained by NeiMeng data by transfer learning. 


Here is another way to build classification model based on pre-trained work. 
```Python 
import torch.nn as nn 

class Classification(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.decoder = nn.Linear(1024, 3) 
    def forward(self, x):
        y = self.decoder(x) 
        return y 
model = PRIMEDP() 
model.load_state_dict(torch.load("ckpt/primedp.middle.pt"))
x = torch.randn([32, 3, 10240])# N, C, T format. 
phase, polar, event_type, wave, hidden = model(x) 

decoder = Classification() 
vector = hidden[:, :, 0]#selected the first feature vecotr. 
vector = vector.detach() 
y = decoder(vector) 
# TODO: define lossfunction and start to train. 
```

### 3. More tasks in PRIME

User can manunally defiend task ID. 
```Python
model = PRIMEDP() 
model.load_state_dict(torch.load("ckpt/primedp.middle.pt"))
# Multi-task ID, 0 event classification, 1 single station location. 
task_id = torch.Tensor([2, 0, 1], dtype=torch.long) 
x = torch.randn([32, 3, 10240])# N, C, T format. 
vect_task, vect_wave = model(x, task_id) 
vect_task=...#Many other task 
```

### 4. Contact
Yuqi Cai: caiyuqiming@foxmail.com 

### LICENSE 
[GPLv3](LICENSE)