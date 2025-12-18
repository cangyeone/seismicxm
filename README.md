# SeismicXM: A Cross-Task Foundation Model for Single-Station Seismic Waveform Processing

SeismicXM is a Transformer-based, cross-task foundation model for **single-station** seismic waveform processing.
It is designed to learn general-purpose seismic representations from large-scale waveform data and to support multiple
downstream tasks in a unified framework, including:

- Seismic phase picking (Pg, Sg, Pn, Sn)
- P-wave first-motion polarity classification
- Event-type classification

![Structure of SeismicXM](fig/structure.en.png)

> Note:
> The project name has been updated from **PRIME-DP** to **SeismicXM** (paper accepted).
> For backward compatibility, some module/class names and checkpoint filenames may still keep the `primedp.*` prefix.

---

## Available Pre-trained Models

| Name | Number of parameters | Path | Status |
|:--:|:--:|:--:|:--:|
| RNN model | 77M | `ckpt/primedp.rnn.pt` | released |
| Picker model | 0.5M | `ckpt/primedp.picker.pt` | released |
| Tiny model | 8.6M | `ckpt/primedp.tinny.pt` | released |
| Middle model | 51M | `ckpt/primedp.middle.pt` | released |
| Event classification model (based on Middle) | 51M | `ckpt/primedp.middle.classification.pt` | released |
| Large model | 1.3B | `ckpt/primedp.large.pt` | training |

---

## 1. Usage

```python
from prime.middle import PRIMEDP
import torch

model = PRIMEDP()

# Load pretrained model
model.load_state_dict(torch.load("ckpt/primedp.middle.pt"))

# Model input: [N, C, T]
x = torch.randn([32, 3, 10240])  # batch, channel, time

# Outputs: phase, polarity, event type, reconstructed waveform, hidden representations
phase, polar, event_type, wave, hidden = model(x)
````

---

## 2. Transfer Learning

Take event classification as an example:

```python
# Define trainable parameters (fine-tune only the event classification decoder)
for key, var in model.named_parameters():
    if var.dtype != torch.float32:
        continue  # e.g., some normalization layers
    if "decoder_event_type" in key:
        var.requires_grad = True
    else:
        var.requires_grad = False

# Define optimizer
optim = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-1
)
```

The weights `ckpt/primedp.middle.classification.pt` are trained on Inner Mongolia data via transfer learning.

Another way to build a classification model based on the pre-trained backbone:

```python
import torch
import torch.nn as nn
from prime.middle import PRIMEDP

class Classification(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Linear(1024, 3)

    def forward(self, x):
        return self.decoder(x)

model = PRIMEDP()
model.load_state_dict(torch.load("ckpt/primedp.middle.pt"))

x = torch.randn([32, 3, 10240])  # [N, C, T]
phase, polar, event_type, wave, hidden = model(x)

decoder = Classification()

# Use the first token / feature vector (project-specific choice)
vector = hidden[:, :, 0].detach()
y = decoder(vector)

# TODO: define loss function and train
```

---

## 3. More Tasks in SeismicXM

Users can manually define `task_id` to switch task heads / embeddings (project-specific mapping).

```python
import torch
from prime.middle import PRIMEDP

model = PRIMEDP()
model.load_state_dict(torch.load("ckpt/primedp.middle.pt"))

# Example task IDs (update according to your implementation):
#   0: event classification
#   1: phase picking
#   2: polarity classification
task_id = torch.tensor([2, 0, 1], dtype=torch.long)

x = torch.randn([32, 3, 10240])  # [N, C, T]
vect_task, vect_wave = model(x, task_id)

vect_task = ...  # task-specific outputs
```

---

## 4. Contact

Yuqi Cai: [caiyuqiming@foxmail.com](mailto:caiyuqiming@foxmail.com)

---

## License & Commercial Use

This project is intended for **scientific research use**.

* **Academic/Research use:** released under **GPLv3** (see [LICENSE](LICENSE)).
* **Commercial use:** please **contact the authors** to obtain a separate commercial license or permission.
