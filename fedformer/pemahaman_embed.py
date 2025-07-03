import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# ── argumen singkat ──────────────────────────
csv = "./dataset/dataset_fed/temperature/temperature_dataset3.csv"
seq_len, BATCH, d_model = 24, 1, 10

# ── 1. load & subset bulan 4-6 ───────────────
df = pd.read_csv(csv, parse_dates=["date"])
train = df[df["date"].dt.month.isin([4,5,6])]

# ambil satu window
num_cols = train.select_dtypes("number").columns[:9]
x = train.iloc[:seq_len][num_cols].values.astype("float32")        # (24,9)

t = train.iloc[:seq_len]["date"]
t_feat = np.stack([
        t.dt.hour/23, t.dt.dayofweek/6, (t.dt.day-1)/30, (t.dt.month-1)/11
    ], axis=-1).astype("float32")                                  # (24,4)

# ── 2. emulate embeddings (NumPy, CPU-only) ─
W_token = np.random.randn(9, d_model).astype("float32")/np.sqrt(9)
W_time  = np.random.randn(4, d_model).astype("float32")/np.sqrt(4)

tok_emb  = x @ W_token              # (24,64)
time_emb = t_feat @ W_time          # (24,64)
embed    = tok_emb + time_emb       # (24,64)

# ── 3. plot  heat-map 10 dim ────────────────
plt.figure(figsize=(8,4))
# plt.imshow(embed[:, :10], aspect="auto", cmap="viridis")
plt.imshow(embed[:, :], aspect="auto", cmap="viridis")
plt.title("DataEmbedding-wo-pos (demo) 24×10 dims")
plt.xlabel("embedding dim 0-9"); plt.ylabel("time-step 0-23")
plt.colorbar(); plt.tight_layout(); plt.show()
