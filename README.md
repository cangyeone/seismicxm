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
    if var.dtype != torch.float32:continue # BN统计计数无梯度
    if "decoder_event_type" in key: # 仅有最后一层有out
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
vector = hidden[:, :, 0]#选择第0个输出，也可以使用其他输出
vector = vector.detach() 
y = decoder(vector) 
# TODO:定义损失函数并对decoder进行训练即可。
```

### 3. 联系方式
有任何方式可以联系：cangye@hotmail.com 