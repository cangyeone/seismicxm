### PRIME地震数据处理模型
我们的模型主要是基于Transformer的，使用Transformer来处理波形的特征。
![structure of PRIME-DP](fig/structure.png)

当前有几个模型
|模型名称|可训练参数数量|模型地址|状态|
|:-:|:-:|:-:|:-:|
|RNN版模型|77M|ckpt/primedp.rnn.pt|发布|
|Picker模型|0.5M|ckpt/primedp.picker.pt|发布|
|小模型|8.6M|ckpt/primedp.tinny.pt|发布|
|中模型|51M|ckpt/primedp.middle.pt|发布|
|大模型|1.3B|ckpt/primedp.large.pt|训练中|

### 1. 使用方式
```Python 
from prime.middle import PRIMEDP 
import torch 

model = PRIMEDP() 
#加载预训练模型
model.load_state_dict(torch.load("ckpt/primedp.middle.pt"))
#输入波形
x = torch.randn([32, 3, 10240])# N, C, T顺序
# 震相, 初动, 地震类型, 波形, 波形特征向量
phase, polar, event_type, wave, hidden = model(x) 
# 可以用于其他处理
# TO:比如使用hidden用于聚类分析
```
