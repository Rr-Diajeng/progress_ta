#!/usr/bin/env python
import argparse, os, re, math, joblib
import pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ────────────────────── CLI ──────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--csv", default="./dataset/dataset_fed/temperature/temperature_dataset3.csv")
p.add_argument("--seq_len", type=int, default=24)
p.add_argument("--split", choices=["train", "val", "test"], default="train")
p.add_argument("--use_scaler", action="store_true")
p.add_argument("--feat_idx", type=int, default=0)
p.add_argument("--kernels", type=str, default="12,24",
               help="daftar kernel MA, pisahkan koma, ex: 12,24,168")
args = p.parse_args()
kernel_sizes = [int(k) for k in args.kernels.split(",")]

# ────────────────────── 1. LOAD CSV ──────────────────────────────────────
df = pd.read_csv(args.csv, parse_dates=["date"])
if "month" not in df.columns:
    df["month"] = df["date"].dt.month

# ubah koma ke titik bila ada
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = pd.to_numeric(df[c].str.replace(",", "."), errors="ignore")

# ────────────────────── 2. SPLIT ala Dataset_Custom ─────────────────────
if args.split == "train":
    df_sel = df[df["month"].isin([4,5,6])]
elif args.split == "val":
    tmp = df[df["month"].isin([6,7])]
    idx = tmp[tmp["month"]==7].index[0]
    df_sel = tmp.loc[idx-args.seq_len:]
else:
    tmp = df[df["month"].isin([7,8])]
    idx = tmp[tmp["month"]==8].index[0]
    df_sel = tmp.loc[idx-args.seq_len:]

if df_sel.empty:
    raise RuntimeError("Subset kosong – cek bulan pada CSV")

num_cols = [c for c in df_sel.select_dtypes(include=["number"]).columns if c!="month"]

# ────────────────────── 3. SCALER (optional) ────────────────────────────
if args.use_scaler:
    scaler = StandardScaler().fit(df[df["month"].isin([4,5,6])][num_cols])
    df_sel.loc[:, num_cols] = scaler.transform(df_sel[num_cols]).astype("float32")
    joblib.dump(scaler, os.path.join(os.path.dirname(args.csv),"std_scaler.pkl"))
    print("[INFO] Scaler saved.")

# ────────────────────── 4. BUAT TENSOR WINDOW ───────────────────────────
window = (df_sel.iloc[:args.seq_len][num_cols]
          .ffill().astype("float32").values)            # (T,C)
x = torch.from_numpy(window).unsqueeze(0)               # (1,T,C)

# ────────────────────── 5. BLOK DEKOMP MULTI ────────────────────────────
class moving_avg(nn.Module):
    def __init__(self, k, stride=1):
        super().__init__()
        self.k=k
        self.avg=nn.AvgPool1d(k, stride=stride, padding=0)
    def forward(self, x):
        front = x[:,:1,:].repeat(1, self.k-1-math.floor((self.k-1)//2), 1)
        back  = x[:,-1:,:].repeat(1, math.floor((self.k-1)//2), 1)
        x_pad = torch.cat([front,x,back],1)
        out   = self.avg(x_pad.permute(0,2,1)).permute(0,2,1)
        return out

class series_decomp_multi(nn.Module):
    def __init__(self, ks):
        super().__init__()
        self.mas = nn.ModuleList([moving_avg(k) for k in ks])
        self.selector = nn.Linear(1, len(ks))
    def forward(self, x):
        means=[]
        for ma in self.mas:
            means.append(ma(x).unsqueeze(-1))           # (B,T,C,1)
        means = torch.cat(means, -1)                    # (B,T,C,K)
        alpha = nn.functional.softmax(self.selector(x.unsqueeze(-1)), -1)
        trend = torch.sum(means*alpha, -1)              # (B,T,C)
        seasonal = x - trend
        return seasonal, trend

seasonal, trend = series_decomp_multi(kernel_sizes)(x)

# ────────────────────── 6. PLOT ─────────────────────────────────────────
idx = args.feat_idx
feat_name = num_cols[idx] if idx < len(num_cols) else f"col{idx}"

for i, col in enumerate(num_cols):
    plt.figure(figsize=(6,3))

    # original
    plt.plot(x[0, :, i].numpy(),                 label="Original", linewidth=2)

    # trend
    plt.plot(trend[0, :, i].detach().numpy(),    label="Trend",    linewidth=2)

    # residual = seasonal
    plt.plot(seasonal[0, :, i].detach().numpy(), label="Residual", linewidth=2)

    plt.title(f"Fitur '{col}'  •  window {args.seq_len} jam")
    plt.xlabel("Time‐step (h)")
    plt.legend(); plt.tight_layout()
    plt.show()
