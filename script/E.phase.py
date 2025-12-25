import torch
from utils.forplt import LargeData2PNWTest
import tqdm
device = torch.device("mps")

dataset = LargeData2PNWTest(balanced=False)
picker1 = torch.jit.load("pickers/seismicxm.02.jit")
picker2 = torch.jit.load("pickers/seist.jit")
picker3 = torch.jit.load("pickers/phasenet.jit")
ofile1 = open("odata/seismicxm.pnw.txt", "w")
ofile2 = open("odata/seist.pnw.txt", "w")
ofile3 = open("odata/phasenet.pnw.txt", "w")

picker1.eval()
picker2.eval() 
picker3.eval()
picker1.to(device)
picker2.to(device)
picker3.to(device)

for i in tqdm.tqdm(range(10000)):
    x, d = dataset.batch_data(1)
    with torch.no_grad():
        x = torch.tensor(x, dtype=torch.float32, device=device)
        x = x.squeeze()
        #print(x.shape)
        p1 = picker1(x)
        p2 = picker2(x[::2])
        p3 = picker3(x)
        p1 = p1.cpu().detach().numpy()
        p2 = p2.cpu().detach().numpy()
        p3 = p3.cpu().detach().numpy()
    ofile1.write(f"#phase,{d[0][0]},{d[0][1]}\n")
    for c, t, p in zip(p1[:, 0], p1[:, 1], p1[:, 2]):
        ofile1.write(f"{c+1},{t},{p}\n")
    ofile2.write(f"#phase,{d[0][0]},{d[0][1]}\n")
    for c, t, p in zip(p2[:, 0], p2[:, 1], p2[:, 2]):
        ofile2.write(f"{c+1},{t*2},{p}\n")
    ofile3.write(f"#phase,{d[0][0]},{d[0][1]}\n")
    for c, t, p in zip(p3[:, 0], p3[:, 1], p3[:, 2]):
        ofile3.write(f"{c+1},{t},{p}\n")