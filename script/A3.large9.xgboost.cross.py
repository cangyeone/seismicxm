from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch 
import numpy as np 



data = torch.load("odata/feature.test.pth", weights_only=False)
N = 176 * 5 
X_test1 = np.array([item[0]["feature_36d"] for item in data], dtype=np.float32)[:N]
d_test1 = np.array([item[1] for item in data], dtype=np.int64)[:N]
e_test1 = [item[2] for item in data][:N]


data = torch.load("odata/feature.pnw.test.pth", weights_only=False)
#print(data[0])
X_test2 = np.array([item[0]["feature_36d"] for item in data], dtype=np.float32)[:5000]
d_test2 = np.array([item[1] for item in data], dtype=np.int64)[:5000]
e_test2 = [item[2] for item in data][:5000]

model1 = XGBClassifier()
model1.load_model("testckpt/xgb_model.json")
model2 = XGBClassifier()
model2.load_model("testckpt/xgb_model.pnw.json")
model3 = XGBClassifier()
model3.load_model("testckpt/xgb_model.pnw.balanced.json")

ofile = open("odata/v9.xgboost.test.pnw.txt", "w")
names = ["xgb", "xgb.pnw", "xgb.pnw.balanced"]
ofiles1 = []
ofiles2 = []
root = "odata/cross/"
for n in names:
    ofiles1.append(open(root+n+".data1.txt", "w"))
    ofiles2.append(open(root+n+".data2.txt", "w"))
ndict = {0:1, 1:0, 2:2}
models = [model1, model2, model3]
for idx, (ofile, model) in enumerate(zip(ofiles1, models)):
    y_test = model.predict(X_test1)
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)
    
    for a1, a2, a3 in zip(y_test, d_test1, e_test1):
        if idx in [1, 2]:
            a1 = ndict[int(a1)]
        ofile.write(f"{a1},{a2},{a3}\n")

for idx, (ofile, model) in enumerate(zip(ofiles2, models)):
    y_test = model.predict(X_test2)
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)
    for a1, a2, a3 in zip(y_test, d_test2, e_test2):
        if idx in [1, 2]:
            a1 = ndict[int(a1)]
            a2 = ndict[int(a2)] 
        else:
            a2 = ndict[int(a2)]
        ofile.write(f"{a1},{a2},{a3}\n")