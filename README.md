
<p align="center">
  <img src="fig/logo.png" alt="SeismicXM logo"/>
</p>

# SeismicXM: A Cross-Task Foundation Model for Single-Station Seismic Waveform Processing

**Code for:**  
- **Title:** *SeismicXM: A Cross-Task Foundation Model for Single-Station Seismic Waveform Processing*  
- **Authors:** Yuqi Cai, Ziye Yu, et al.  
- **Contact:** Ziye Yu ([yuziye@cea-igp.ac.cn](mailto:yuziye@cea-igp.ac.cn))

---

## Overview

**SeismicXM** is a Transformer-based, cross-task foundation model for **single-station seismic waveform processing**.  
It is designed to learn general-purpose seismic representations from large-scale waveform data and to support multiple downstream tasks within a unified framework, including:

- Seismic phase picking (Pg, Sg, Pn, Sn)
- P-wave first-motion polarity classification
- Event-type classification

The model follows a shared-backbone, multi-head design, enabling efficient transfer learning and task adaptation across different regions and datasets.

![Structure of SeismicXM](fig/structure.en.png)

> **Note**  
> The project name has been officially updated from **PRIME-DP** to **SeismicXM** following paper acceptance.  
> Some legacy checkpoints or internal identifiers may still reflect earlier naming conventions.

---

## 1. Basic Usage

```python
import torch
from seismicxm.middle import SeismicXM

# Initialize model
model = SeismicXM()

# Load pretrained weights
model.load_state_dict(torch.load("ckpt/seismicxm.middle.pt"))

# Model input: [N, C, T]
x = torch.randn([32, 3, 10240])  # batch, channel, time

# Outputs:
# phase picking, polarity, event type, reconstructed waveform, hidden representations
phase, polar, event_type, wave, hidden = model(x)
````

---

## 2. Transfer Learning

Taking **event-type classification** as an example, one may fine-tune only the corresponding task head while freezing the shared backbone.

```python
# Enable gradients only for the event classification decoder
for name, param in model.named_parameters():
    if param.dtype != torch.float32:
        continue
    if "decoder_event_type" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Optimizer
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-1
)
```

The checkpoint `ckpt/seismicxm.middle.classification.pt` is obtained by transfer learning on regional seismic data.

---

### Alternative: External Task Head

Another common strategy is to attach a lightweight task-specific head on top of the pretrained backbone.

```python
import torch
import torch.nn as nn
from seismicxm.middle import SeismicXM

class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 3)

    def forward(self, x):
        return self.fc(x)

model = SeismicXM()
model.load_state_dict(torch.load("ckpt/seismicxm.middle.pt"))

x = torch.randn([32, 3, 10240])
phase, polar, event_type, wave, hidden = model(x)

# Use the first token / feature vector (project-specific choice)
feature = hidden[:, :, 0].detach()

decoder = ClassificationHead()
y = decoder(feature)
```

---

## 3. Multi-Task Control

SeismicXM supports task-aware embeddings through a manually specified `task_id`.

```python
import torch
from seismicxm.middle import SeismicXM

model = SeismicXM()
model.load_state_dict(torch.load("ckpt/seismicxm.middle.pt"))

# Example task IDs:
# 0: event classification
# 1: phase picking
# 2: polarity classification
task_id = torch.tensor([2, 0, 1], dtype=torch.long)

x = torch.randn([32, 3, 10240])
vect_task, vect_wave = model(x, task_id)
```

---

## 4. Model Zoo

This repository provides pretrained and fine-tuned models for seismic phase picking, event detection, and downstream classification within the **SeismicXM / PnSn** framework.

Models differ in training strategy, dataset domain, class balancing, and number of training iterations.

### Naming Convention

```
<strategy>[.<domain>][.<balancing>].<iterations>[.<dataset>][.<type>].pt
```

* **strategy**

  * `original` – trained from scratch
  * `transfer` – fine-tuned from a pretrained base model
  * `seismicxm.base` – general-purpose foundation model
  * `nowave` – ablation model without waveform information

* **domain**

  * `pnw` – Pacific Northwest dataset

* **balancing**

  * `balanced` – class-balanced training

* **iterations**

  * `100`, `200`, `5000` – number of training iterations

* **dataset**

  * `mine` – in-house dataset

* **type**

  * `enc` – encoder-only model for feature extraction

---

## 5. Deployment and Auxiliary Models

### TorchScript Model

* **`model.all.jit`**
  Fully exported TorchScript model for efficient deployment.

**Use cases**

* Production inference
* C++ / service-side deployment
* Faster loading and reduced runtime dependencies

---

### XGBoost Models

XGBoost models are typically used in **two-stage pipelines**, where SeismicXM encoders provide feature embeddings.

* `xgb_model.json` – general domain
* `xgb_model.pnw.json` – PNW-specific
* `xgb_model.pnw.balanced.json` – balanced PNW data

---

## 6. Recommended Model Selection

* **Fast deployment**
  → `model.all.jit`

* **New region / station network**
  → `seismicxm.base.pt` + transfer learning

* **PNW applications**
  → `transfer.pnw.balanced.200.pt`

* **Operational / in-house data**
  → `transfer.balanced.200.mine.pt`

* **Feature-based or interpretable pipelines**
  → `*.enc.pt` + `xgb_model*.json`

---

## Contact

* **Yuqi Cai:** [caiyuqiming@foxmail.com](mailto:caiyuqiming@foxmail.com)
* **Ziye Yu:** [yuziye@cea-igp.ac.cn](mailto:yuziye@cea-igp.ac.cn)

---

## License & Commercial Use

This project is intended for **scientific research use**.

* **Academic / Research use:** released under **GPLv3** (see [LICENSE](LICENSE))
* **Commercial use:** please contact the authors to obtain a separate commercial license or permission
