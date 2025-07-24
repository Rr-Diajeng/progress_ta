import streamlit as st
import pandas as pd
import numpy as np
import torch
from datetime import timedelta

SEQ_LEN      = 24                    
LABEL_LEN    = 24                    
PRED_LEN     = 1          
FEATURE_COLS = ["TEMPERATURE", "DEW_POINT", "PRECIPITABLE WATER", "GHI", "PRESSURE", "OT"]
MODEL_PATH   = "./interface/checkpoint/checkpoint_corrkw.pt"

@st.cache_resource
def load_model(path=MODEL_PATH):
    mdl = torch.jit.load(path, map_location="cpu")
    mdl.eval()
    return mdl

def time_features(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.stack([
        idx.month       / 12.0,
        idx.day         / 31.0,
        idx.dayofweek   / 7.0,
        idx.hour        / 24.0,
    ], axis=1).astype(np.float32)

def build_tensors(win_df: pd.DataFrame, target_ts: pd.Timestamp):

    arr = win_df[FEATURE_COLS].to_numpy(np.float32)
    z   = (arr - X_MEAN) / X_STD

    # 2. buat x_enc, x_dec
    x_enc = z
    dummy = np.zeros((1, len(FEATURE_COLS)), np.float32)
    x_dec = np.concatenate([x_enc, dummy], axis=0)

    # 3. timestamp mark
    x_mark_enc = time_features(win_df.index)
    x_mark_dec = np.concatenate([x_mark_enc,
                                 time_features(pd.DatetimeIndex([target_ts]))])

    return (torch.tensor(x_enc[None]),      # (1,24,6)
            torch.tensor(x_mark_enc[None]),
            torch.tensor(x_dec[None]),      # (1,25,6)
            torch.tensor(x_mark_dec[None]))


meta = np.load("./interface/checkpoint/scaler_meta_corrkw.npz", allow_pickle=True)
X_MEAN, X_STD = meta["x_mean"].astype(np.float32), meta["x_std"].astype(np.float32)
OT_MEAN, OT_STD = float(meta["y_mean"]), float(meta["y_std"])
TRAIN_COLS = meta["feature_cols"].tolist()          # urutan aslinya
assert TRAIN_COLS == FEATURE_COLS, "Urutan kolom tidak sama dg training!"

@torch.no_grad()
def predict_one(model, win_df, target_ts):
    xe, xme, xd, xmd = build_tensors(win_df, target_ts)
    z = model(xe, xme, xd, xmd).squeeze()
    if z.numel() > 1: z = z[-1]
    return float(z.item()*OT_STD + OT_MEAN)

st.title("📈 Dashboard Prediksi OT/KW pada Selected Fitur - FEDformer WEB")

file = st.file_uploader(
    "📂 Upload CSV",
    type="csv"
)


if st.button("🚀 Prediksi"):
    if file is None:
        st.warning("Silakan unggah CSV terlebih dulu.")
        st.stop()

    # ―― Load & cek CSV ――
    df = pd.read_csv(file)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop(columns=["date", "time"]).set_index("datetime").sort_index()

    miss = [c for c in FEATURE_COLS if c not in df.columns]
    if miss:
        st.error(f"Kolom hilang di CSV: {miss}")
        st.stop()

    model = load_model()

    year  = df.index[0].year
    start = pd.Timestamp(year=year, month=8, day=1, hour=0)
    end   = pd.Timestamp(year=year, month=8, day=31, hour=23)

    preds, times = [], []
    with st.spinner("⏳ Memproses prediksi Agustus…"):
        t = start
        while t <= end:
            win_df = df.loc[t - timedelta(hours=SEQ_LEN) : t - timedelta(hours=1)]
            if len(win_df) == SEQ_LEN:
                preds.append(predict_one(model, win_df, t))
                times.append(t)
            t += timedelta(hours=1)

    if preds:
        out = pd.DataFrame({"timestamp": times, "OT_pred": preds}).set_index("timestamp")
        st.success(f"✅ {len(out)} prediksi OT untuk Agustus {year}.")
        st.dataframe(out)
        st.line_chart(out)
    else:
        st.error("Data historis 24 jam sebelum 1 Agustus belum lengkap di CSV.")