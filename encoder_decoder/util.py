import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


def fitModel(batch_size, steps_per_epoch, x_scaled,
             y_scaled):
    
    while True:
        for _ in range(steps_per_epoch):
            encoder_input = x_scaled
            decoder_output = y_scaled
            decoder_input = np.zeros((decoder_output.shape[0], decoder_output.shape[1], 1))

            yield((encoder_input, decoder_input), decoder_output)

def plot_prediction(x, y_true, y_pred):
    plt.figure(figsize=(12, 3))
    # Ambil jumlah output dimensi berdasarkan dimensi x jika dimensi tidak tersedia di y_true dan y_pred
    output_dim = x.shape[-1] if len(y_true.shape) == 1 else y_true.shape[1]

    for j in range(output_dim):
        past = x[:, j]
        # Jika y_true dan y_pred hanya memiliki satu dimensi, gunakan tanpa indeks kedua
        true = y_true[:, j] if len(y_true.shape) > 1 else y_true
        pred = y_pred[:, j] if len(y_pred.shape) > 1 else y_pred

        label1 = "Seen (past) values" if j == 0 else "_nolegend_"
        label2 = "True future values" if j == 0 else "_nolegend_"
        label3 = "Predictions" if j == 0 else "_nolegend_"

        plt.plot(range(len(past)), past, "o--b", label=label1)
        plt.plot(range(len(past), len(true) + len(past)), true, "x--b", label=label2)
        plt.plot(range(len(past), len(pred) + len(past)), pred, "o--y", label=label3)
    
    plt.legend(loc='best')
    plt.title("Predictions vs. True Values")
    plt.show()