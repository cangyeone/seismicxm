import torch
import torch.nn as nn

from seismicxm import SeismicXM


class Classification(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Linear(1024, 3)

    def forward(self, x):
        return self.decoder(x)


model = SeismicXM()
model.load_state_dict(torch.load("ckpt/seismicxm.middle.pt"))

x = torch.randn([32, 3, 10240])  # N, C, T format.
phase, polar, event_type, wave, hidden = model(x)

decoder = Classification()
vector = hidden[:, :, 0]  # 选择第0个输出，也可以使用其他输出
vector = vector.detach()
y = decoder(vector)
# TODO:定义损失函数即可
