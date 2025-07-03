# install pytorch (CPU) jika belum
# pip install torch --index-url https://download.pytorch.org/whl/cpu

import torch, pandas as pd, math
import matplotlib.pyplot as plt
from torch import nn

# ─── 1. Siapkan data window 24 jam (bulan 4–6) ───
df = pd.read_csv("./dataset/dataset_fed/temperature/temperature_dataset3.csv", parse_dates=["date"])
win = df[df.date.dt.month.isin([4,5,6])].iloc[:24]

num_cols = win.select_dtypes("number").columns[:9]
x = torch.tensor(win[num_cols].values, dtype=torch.float32).unsqueeze(0)     # (1,24,9)

tfeat = torch.tensor(
    pd.DataFrame({
        'hour':  win.date.dt.hour/23,
        'dow':   win.date.dt.dayofweek/6,
        'day':   (win.date.dt.day-1)/30,
        'month': (win.date.dt.month-1)/11
    }).values, dtype=torch.float32).unsqueeze(0)                              # (1,24,4)

# ─── 2. Definisi layer persis FEDformer ───
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, 3, padding=1, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, z):                          # (B,T,C)
        return self.conv(z.permute(0,2,1)).permute(0,2,1)

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model): super().__init__(); self.lin = nn.Linear(4, d_model, bias=False)
    def forward(self, m): return self.lin(m)       # (B,T,4) → (B,T,512)

class DataEmbed(nn.Module):
    def __init__(self, c_in=9, d_model=512): super().__init__()
    def __init__(self, c_in=9, d_model=512):
        super().__init__()
        self.val = TokenEmbedding(c_in, d_model)
        self.time = TimeFeatureEmbedding(d_model)
    def forward(self,x,xm): return self.val(x)+self.time(xm)

embed = DataEmbed(c_in=9, d_model=512)

with torch.no_grad():
    enc_emb = embed(x, tfeat)    # (1,24,512)

print("shape:", enc_emb.shape)   # → (1,24,512)

# ─── 3. FFT low-modes demo (modes=4) ───
x_h = enc_emb.view(1,24,8,64).permute(0,2,3,1)     # (1,8,64,24)
X = torch.fft.rfft(x_h, dim=-1)                    # (1,8,64,13)
sel = [0,1,2,3]                                    # simpan 4 mode
mask = torch.zeros_like(X); mask[...,sel] = 1
X_low = X*mask
x_low = torch.fft.irfft(X_low, n=24, dim=-1).permute(0,3,1,2).reshape(1,24,512)

plt.plot(enc_emb[0,:,0].numpy(), label="orig dim0")
plt.plot(x_low[0,:,0].numpy(), label="after FFT 4 modes")
plt.legend(); plt.show()
