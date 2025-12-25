import torch  
from models.EQLarge9 import EQLargeCNN 
import torch.nn as nn 
from typing import Any, Dict

class ModelJIT(nn.Module):
    __annotations__: Dict[str, Any] = {}
    def __init__(self):
        super().__init__()
        self.model = EQLargeCNN()
    def forward(self, x):
        x -= torch.mean(x, dim=2, keepdim=True)
        x /= x.abs().max(dim=2, keepdim=True)[0]
        phase, polar, event_type, wave, mu, logvar, _ = self.model(x)
        return phase, polar.softmax(dim=1), event_type.softmax(dim=-1)
    



model = ModelJIT() 
model.eval()
ipth = torch.load('testckpt/transfer.balanced.200.mine.enc.pt')
opth = {}
for k, v in ipth.items():
    opth["model."+k] = v 
model.load_state_dict(opth)

model_jit = torch.jit.script(model)
torch.jit.save(model_jit, 'testckpt/model.all.jit')

x = torch.randn([10, 3, 3072])
phase, polar, event_type = model_jit(x)
print(phase.shape, polar.shape, event_type.shape)
input_names = ["wave"]
output_names = ["phase", "polar", "etype"]
torch.onnx.export(model, x, 
"pickers/mine.all.onnx", verbose=True, 
dynamic_axes={"wave":{0:"batch", 2:"length"}, "prob":{0:"batch", 2:"length"}, "time":{0:"batch"}}, 
input_names=input_names, output_names=output_names, opset_version=19)