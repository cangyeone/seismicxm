### PRIME地震数据处理模型
我们的模型主要是基于Transformer的，使用Transformer来处理波形的特征。
![structure of PRIME-DP](fig/structure.png)

当前有几个模型
|模型名称|可训练参数数量|模型地址|状态|
|:-:|:-:|:-:|:-:|
|RNN版模型|77M|ckpt/primedp.rnn.pt|发布|
|小模型|8.6M|ckpt/primedp.tinny.pt|发布|
|中模型|51M|ckpt/primedp.middle.pt|发布|
|大模型|1.3B|ckpt/primedp.large.pt|训练中|


