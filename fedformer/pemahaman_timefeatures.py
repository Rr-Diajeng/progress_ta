import pandas as pd
import numpy as np

def time_features(dates, freq='h'):
    """Return matrix shape (n_features, n_timestamps)"""
    class HourOfDay:
        def __call__(self, idx):
            return idx.hour / 23.0 - 0.5
    class DayOfWeek:
        def __call__(self, idx):
            return idx.dayofweek / 6.0 - 0.5
    class DayOfMonth:
        def __call__(self, idx):
            return (idx.day - 1) / 30.0 - 0.5
    class DayOfYear:
        def __call__(self, idx):
            return (idx.dayofyear - 1) / 365.0 - 0.5

    return np.vstack([
        HourOfDay()(dates),
        DayOfWeek()(dates),
        DayOfMonth()(dates),
        DayOfYear()(dates)
    ])

# === PARAMETER DATASET ===
start_train = pd.Timestamp("2019-04-01 00:00")
seq_len, label_len, pred_len = 24, 24, 1   # sesuai dataset

def show_window(start_time, window_id):
    """Cetak info seq_x_mark & seq_y_mark (dengan & tanpa .T) untuk window tertentu."""
    print(f"\n=== WINDOW {window_id} ===")
    # Geser start_time sesuai window
    win_start = start_time + pd.Timedelta(hours=window_id)

    # seq_x_mark
    dates_x = pd.date_range(win_start, periods=seq_len, freq='H')
    mat_x_noT = time_features(dates_x, freq='h')
    mat_x_T   = mat_x_noT.T

    print(f"seq_x_mark shape tanpa .T  : {mat_x_noT.shape}")
    print(f"seq_x_mark shape dengan .T : {mat_x_T.shape}")

    df_x_T = pd.DataFrame(mat_x_T, columns=['HourOfDay','DayOfWeek','DayOfMonth','DayOfYear'])
    df_x_T['Datetime'] = dates_x

    # seq_y_mark
    dates_y = pd.date_range(win_start, periods=label_len + pred_len, freq='H')
    mat_y_noT = time_features(dates_y, freq='h')
    mat_y_T   = mat_y_noT.T

    print(f"seq_y_mark shape tanpa .T  : {mat_y_noT.shape}")
    print(f"seq_y_mark shape dengan .T : {mat_y_T.shape}")

    df_y_T = pd.DataFrame(mat_y_T, columns=['HourOfDay','DayOfWeek','DayOfMonth','DayOfYear'])
    df_y_T['Datetime'] = dates_y

    # Tampilkan 3 baris pertama & terakhir
    print("\nseq_x_mark (3 head & 3 tail) setelah .T:")
    print(df_x_T.head(3))
    print(df_x_T.tail(3))

    print("\nseq_y_mark (3 head & 3 tail) setelah .T:")
    print(df_y_T.head(3))
    print(df_y_T.tail(3))

# Tampilkan window 0‑2
for w in range(3):
    show_window(start_train, w)
