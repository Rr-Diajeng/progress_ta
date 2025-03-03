import os
import time

import pandas as pd
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (8, 5.2)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

#%%

def check_folder_exist(path, create_flag=True, show_flag=False):
    if not os.path.exists(path):
        print("Folder not exist.", end=" ") 
        if create_flag:
            print("Create new folder")
            new_path = create_folder(path)
            return new_path
        else:
            print("create flag is False, will not create new folder")
            return None
    else:
        print("Folder " + path + " exist")
        # show subfolder and files
        if show_flag:
            for root, dirs, files in os.walk(path):
                level = root.replace(path, '').count(os.sep)
                indent = ' ' * 4 * (level)
                print('{}{}/'.format(indent, os.path.basename(root)))
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    print('{}{}'.format(subindent, f))
        return path + '/'
    
def find_occur(text_str, find_str):
    res = [i for i in range(len(text_str)) if text_str.startswith(find_str, i)]
    return res

def create_folder(path, img=False):
    Path(path).mkdir(parents=True, exist_ok=True)
    if img:
        path_img = path + '/image' 
        Path(path_img).mkdir(parents=True, exist_ok=True)
        return path + '/', path_img + '/'
    else:
        return path + '/'
    
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_period=False):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_period:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def plot_metrics(history, metrics=['loss'], title='Metrics', path=None, show=True):
    plt.figure(figsize=(8, 5))
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        # plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[3], linestyle="--", label='Val')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(name, fontsize=12)
        plt.tight_layout()
        plt.legend()
    plt.suptitle(title.lower(), fontsize=12, y=1.02)
    if path:    
        plt.savefig(path +  title.lower() + '.png', bbox_inches="tight")
    if show: plt.show()
    plt.close()

METRICS = [tf.keras.metrics.MeanSquaredError(name='mse'),
           # tf.keras.metrics.MeanAbsoluteError(name='mae'), 
           # tf.keras.metrics.MeanAbsolutePercentageError(name='mape')
           ]

def model_lstm(n_shape, n_output, n_hidden=32):
    model  = tf.keras.models.Sequential()
    inputs = tf.keras.layers.Input(shape=(n_shape,1), name='input')
    # flattn = tf.keras.layers.Flatten()(inputs)
    hidden = tf.keras.layers.LSTM(units=n_hidden, 
                                  activation='tanh',
                                  recurrent_activation='sigmoid',
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                  name='hidden')(inputs)
    # hidden = tf.keras.layers.Dropout(0.2)(hidden)
    output = tf.keras.layers.Dense(units=n_output, name='output')(hidden)
    model  = tf.keras.Model(inputs=[inputs], outputs=[output])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(
                    # lr=0.001, 
                    # epsilon=0.01
                    ),
                metrics=METRICS)
    return model

def denormalized(y_true_scaled, y_pred_scaled, scaler):
    y_true = np.transpose(scaler.inverse_transform(np.transpose(y_true_scaled)))
    y_pred = np.transpose(scaler.inverse_transform(np.transpose(y_pred_scaled))) 
    return np.round(y_true,2), np.round(y_pred,2)

def get_result(y_true_list, y_pred_list):
    true_sum = np.zeros((y_pred_list[0].shape[0], y_pred_list[0].shape[1]))
    pred_sum = np.zeros((y_pred_list[0].shape[0], y_pred_list[0].shape[1]))
    for i in range(len(y_pred_list)):
        true_sum = true_sum + y_true_list[i]
        pred_sum = pred_sum + y_pred_list[i]
    
    return true_sum, pred_sum

def evaluate_metric(true, pred):
    mae_list = list()
    mse_list = list()
    rmse_list = list()
    mape_list = list()
    for i in range(true.shape[0]):
        mae  = mean_absolute_error(true[i], pred[i])
        mse  = mean_squared_error(true[i], pred[i])
        rmse = mean_squared_error(true[i], pred[i], squared=False)
        mape = mean_absolute_percentage_error(true[i], pred[i])
        mae_list.append(round(mae, 3))
        mse_list.append(round(mse, 3))
        rmse_list.append(round(rmse, 3))
        mape_list.append(round(mape, 3))

    print("mae :", round(np.mean(mae_list), 3))
    print("mse :", round(np.mean(mse_list), 3))
    print("rmse:", round(np.mean(rmse_list), 3))
    if round(np.mean(mape_list), 3) > 1:
        print("mape is not applicable")
    else:    
        print("mape:", round(np.mean(mape_list), 3))

    return np.array(mae_list), np.array(mse_list), np.array(rmse_list), np.array(mape_list)

def find_index_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_result(true, pred, list, input_width, target_width, single_period, title, path):
    
    if single_period:
        idx = np.random.randint(0, true.shape[0]-INPUT_WIDTH)
        x = range(idx, idx+target_width)


        plt.figure(figsize=(8, 4))
        plt.plot(x, true[idx:idx+input_width].ravel(), linestyle="-", color="#2ca02c",
                 marker="o", markersize=8, markeredgecolor="k")
        ax = plt.plot(x, pred[idx:idx+input_width].ravel(), linestyle="-", color="#ff7f0e", 
                      marker="X", markersize=8,markeredgecolor="k")
        plt.title(title.upper() + ": " + str(np.mean(list[x]).round(3)), fontsize=12)

    else:
        indices = [find_index_nearest(list, np.mean(list)),
                np.where(list == np.min(list))[0][0], 
                np.where(list == np.max(list))[0][0]
                    ]
        index_names = ["Average", "Best", "Worst"]
        
        x = range(0, target_width)
        xticks = range(0, target_width, 4)
        xlabels = ["t+"+str(i) for i in xticks]

        fig, ax = plt.subplots(3,1, sharex=True)
        fig.set_size_inches(8, 8)
        for i, (idx, name) in enumerate(zip(indices, index_names)):
            ax[i].set_ylabel("Power (kW)", fontsize=12)
            ax[i].set_title(title.upper() + ": " + str(list[indices[i]]) + " ["+name+"]", fontsize=12)

            ax[i].plot(x, true[idx], color="#2ca02c", linestyle="--")
            ax[i].scatter(x, true[idx], edgecolors='k', c="#2ca02c", s=64, marker="o", label='True')
            ax[i].scatter(x, pred[idx], edgecolors='k', c="#ff7f0e", s=64, marker="X", label='Pred')
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels(xlabels)

            if i == 0:
                ax[0].legend(loc=0)

        fig.set_tight_layout(True)

    plt.xlabel("Time (h)", fontsize=12)
    if path is not None:
        plt.savefig(path + "_" + title.lower().replace(" ", "_") + ".png", bbox_inches="tight")
    plt.show()
    plt.close()

from keras.layers import Input, LSTM, Dense, Concatenate, Conv1D, MaxPooling1D, Reshape
from keras.models import Model
from keras import layers, utils, callbacks

def build_model(n_input, n_output, num_features):
    tf.keras.backend.clear_session() 
    
    # Encoder Input
    encoder_inputs = Input(shape=(n_input, num_features))
    encoder_lstm = encoder_inputs   

    # encoder_lstm = layers.LayerNormalization(epsilon=1e-6)(encoder_inputs)
    # # encoder_lstm = LSTM(32, return_sequences=True)(encoder_lstm)
    # encoder_lstm, state_h, state_c = LSTM(32, 
    #                                       kernel_initializer='he_normal',
    #                                       kernel_regularizer=tf.keras.regularizers.l2(0.01),
    #                                       return_state=True, return_sequences=True)(encoder_lstm)

    # # Decoder LSTM with initial state from encoder LSTM

    # decoder_lstm = LSTM(32, return_sequences=True, name="decoder_lstm_1")(encoder_lstm, initial_state=[state_h, state_c])
    # decoder_lstm = LSTM(32, return_sequences=False, name="decoder_lstm_2")(decoder_lstm)

    # decoder_lstm = layers.LayerNormalization(epsilon=1e-6)(decoder_lstm)

    # for dim in [24]:
    #     decoder_lstm = layers.Dense(dim, activation="relu")(decoder_lstm)
    #     decoder_lstm = layers.Dropout(0.2)(decoder_lstm)       


    # Encoder LSTM
    encoder_lstm = layers.LayerNormalization(epsilon=1e-6)(encoder_lstm) 
    encoder_lstm = LSTM(units=32, name="lstm_encoder_hidden", return_sequences=True, 
                        kernel_initializer='he_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01))(encoder_lstm)
    encoder_lstm = layers.Dropout(0.25)(encoder_lstm)
    encoder_lstm, state_h, state_c = LSTM(units=32, return_state=True, return_sequences=True, 
                                          name="lstm_encoder")(encoder_lstm)

    # Decoder LSTM with initial state from encoder LSTM   
    decoder_lstm = LSTM(units=32, return_sequences=True, name="lstm_decoder")(encoder_lstm, initial_state=[state_h, state_c])    
    decoder_lstm = layers.LayerNormalization(epsilon=1e-6)(decoder_lstm)
    # decoder_lstm = LSTM(units=168, name="lstm_decoder_hidden", return_sequences=True)(decoder_lstm)
    decoder_lstm = layers.Dropout(0.2)(decoder_lstm)
    decoder_lstm = LSTM(units=32, name="lstm_decoder_final", return_sequences=False)(decoder_lstm)    
    
    for dim in [24]:
        decoder_lstm = layers.Dense(dim, activation="relu")(decoder_lstm)
        decoder_lstm = layers.Dropout(0.2)(decoder_lstm)   

    outputs = layers.Dense(n_output)(decoder_lstm)

    # Model
    model = Model(encoder_inputs, outputs)    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.01), loss='mse')

    return model

#%%
# decomposition_name = "ewt"
forecasting_name = "encoder_decoder"
method_name = forecasting_name          # decomposition_name + "_" + forecasting_name
location_name = "renwu"
target_name = "power"

cv_names = ["cv1", "cv2", "cv3", "cv4", "cv5"]
# cv_names = cv_names[0:1]

train_cv_datetimes = np.array([["2020-01-01 00:00:00", "2020-12-31 23:59:59"],
                               ["2020-01-01 00:00:00", "2021-03-31 23:59:59"],
                               ["2020-01-01 00:00:00", "2021-06-30 23:59:59"],
                               ["2020-01-01 00:00:00", "2021-09-30 23:59:59"],
                               ["2020-01-01 00:00:00", "2021-12-31 23:59:59"]]) 
test_cv_datetimes  = np.array([["2021-01-01 00:00:00", "2021-03-31 23:59:59"],
                               ["2021-04-01 00:00:00", "2021-06-30 23:59:59"],
                               ["2021-07-01 00:00:00", "2021-09-30 23:59:59"],
                               ["2021-10-01 00:00:00", "2021-12-31 23:59:59"],
                               ["2022-01-01 00:00:00", "2022-12-31 23:59:59"]])


#%%
# Parameter for data setting
INPUT_WIDTH  = 168
TARGET_WIDTH = 24
STEP = 1
SINGLE_PERIOD = False

if SINGLE_PERIOD: 
    TARGET_WIDTH = 1
    period_str = "single_period"
else: period_str = "multi_period"

# Parameter for model
MAX_EPOCHS = 2
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.25

print("Input width  : ", INPUT_WIDTH)
print("Target width : ", TARGET_WIDTH)
print("Step         : ", STEP)
print("Single period: ", SINGLE_PERIOD)
print("")

path = "result/" + method_name + "/" + location_name + "/target_width_" + str(TARGET_WIDTH)
result_directory = check_folder_exist(path, create_flag=True, show_flag=False)

#%%
df_evaluations = pd.DataFrame()
print("Cross Validation for", location_name, "dataset")
for i in range(len(cv_names)):
    # i = 0

    result_directory_cv = check_folder_exist(path = result_directory + cv_names[i], create_flag=True, show_flag=False)     
    
    result_file_name_cv = method_name + "_" + location_name + "_" + target_name + "_" + period_str + "_" + cv_names[i]
    result_file_path_cv = result_directory_cv + result_file_name_cv    
    

    print("  CV    : ", cv_names[i])
    print("  Train : ", train_cv_datetimes[i])
    print("  Test  : ", test_cv_datetimes[i])

    train_start_datetime = str(train_cv_datetimes[i, 0])
    train_end_datetime   = str(train_cv_datetimes[i, 1])
    test_start_datetime  = str(test_cv_datetimes[i, 0])
    test_end_datetime    = str(test_cv_datetimes[i, 1])

    df = pd.read_csv("dataset/transformed/" + location_name + "_transformed.csv", 
                    index_col=0, parse_dates=True)
    df = df.loc[df.index <= test_end_datetime]
    cols = df.columns

    print("  Dataset Last index : ", df.index[-1])

    y_train_pred_list = list()
    y_test_pred_list  = list()
    y_train_true_list = list()
    y_test_true_list  = list()

    print("")

    feature_selection = pd.read_excel("dataset/feature_selection/feature_selection.xlsx", index_col=0)
    feature_selection = feature_selection.loc[(feature_selection["location"] == location_name) & (feature_selection["method"] == "xgb") & (feature_selection["target"] == target_name)]
    feature_names = feature_selection.loc[feature_selection["test_mae"] == feature_selection["test_mae"].min(), "feature_selected"].apply(eval).to_list()[0]  

    df_features = df[feature_names].copy()
    df_target = df[target_name].copy()

    df_target_train = df_target.loc[(df_target.index >= train_start_datetime) & (df_target.index <= train_end_datetime)]
    df_target_test  = df_target.loc[(df_target.index >= test_start_datetime) & (df_target.index <= test_end_datetime)]

    scaler_target = StandardScaler()
    scaler_target.fit(df_target_train.to_numpy().reshape(-1, 1))
    train_target_scaled = scaler_target.transform(df_target_train.to_numpy().reshape(-1, 1))
    test_target_scaled  = scaler_target.transform(df_target_test.to_numpy().reshape(-1, 1))

    df_feature_train = df_features.loc[(df_features.index >= train_start_datetime) & (df_features.index <= train_end_datetime)]
    df_feature_test  = df_features.loc[(df_features.index >= test_start_datetime) & (df_features.index <= test_end_datetime)]

    scaler_feature = StandardScaler()
    scaler_feature.fit(df_feature_train.to_numpy())
    train_feature_scaled = scaler_feature.transform(df_feature_train.to_numpy())
    test_feature_scaled  = scaler_feature.transform(df_feature_test.to_numpy())

    # Concatenating energy and other features 
    train_all_features_scaled = np.concatenate([train_target_scaled, train_feature_scaled], axis=1)                                      
    test_all_features_scaled  = np.concatenate([test_target_scaled, test_feature_scaled], axis=1)

    X_train_all_features_scaled, y_train_target_scaled = multivariate_data(train_all_features_scaled, train_target_scaled, 
                                                                           start_index=0, end_index=None, 
                                                            history_size=INPUT_WIDTH, 
                                                            target_size=TARGET_WIDTH, 
                                                            step=STEP, single_period=SINGLE_PERIOD)
    X_test_all_features_scaled, y_test_target_scaled   = multivariate_data(test_all_features_scaled, test_target_scaled, 
                                                            start_index=0, end_index=None, 
                                                            history_size=INPUT_WIDTH, 
                                                            target_size=TARGET_WIDTH, 
                                                            step=STEP, single_period=SINGLE_PERIOD)


    y_train_target_scaled = y_train_target_scaled.reshape(-1, TARGET_WIDTH)
    y_test_target_scaled  = y_test_target_scaled.reshape(-1, TARGET_WIDTH)


    weight_file_path = result_file_path_cv + "_weight.h5"
    model_checkpoint_callback = callbacks.ModelCheckpoint(monitor='val_loss', 
                                                          mode='min',
                                                          filepath=weight_file_path,
                                                          save_best_only=True)
    callback_list = [callbacks.EarlyStopping(patience=10, restore_best_weights=True), 
                     model_checkpoint_callback
                    ] 

    model = build_model(n_input=INPUT_WIDTH, n_output=TARGET_WIDTH, num_features=X_train_all_features_scaled.shape[2])            
    
    start_time = time.time()
    
    history = model.fit(X_train_all_features_scaled, y_train_target_scaled, 
                        # shuffle=False,
                        epochs=MAX_EPOCHS, 
                        batch_size=32, 
                        validation_split=VALIDATION_SPLIT,
                        callbacks=callback_list
                        )

    y_train_pred_scaled = model.predict(X_train_all_features_scaled)
    y_test_pred_scaled = model.predict(X_test_all_features_scaled)
    
    execution_time = round(time.time() - start_time, 3)

    metrics_title_cv = result_file_name_cv + "_metrics"
    metrics_file_path_cv = result_directory_cv
    plot_metrics(history, title=metrics_title_cv, show=True, path=metrics_file_path_cv)    
                    
    # y_test_true, y_test_pred = denormalized(y_test_target_scaled, y_test_pred_scaled, scaler_target)
    # test_mae_list, test_mse_list, test_rmse_list, test_mape_list = evaluate_metric(y_test_true, y_test_pred)

    # y_test_pred = np.zeros((y_test_pred_scaled.shape[0], y_test_pred_scaled.shape[1]))
    # for i in range(len(y_test_pred_scaled)):
    #     y_test_pred[i] = np.transpose(scaler_target.inverse_transform(y_test_pred_scaled[i].reshape(-1,1)))

    # y_test_true = np.zeros((y_test_target_scaled.shape[0], y_test_target_scaled.shape[1]))
    # for i in range(len(y_test_target_scaled)):
    #     y_test_true[i] = np.transpose(scaler_target.inverse_transform(y_test_target_scaled[i].reshape(-1,1)))

    # test_mae_list, test_mse_list, test_rmse_list, test_mape_list = evaluate_metric(y_test_true, y_test_pred)


    print("Train evaluation:")
    y_train_true, y_train_pred = denormalized(y_train_target_scaled, y_train_pred_scaled, scaler_target)
    train_mae_list, train_mse_list, train_rmse_list, train_mape_list = evaluate_metric(y_train_true, y_train_pred)

    print("\nTest evaluation:")
    y_test_true, y_test_pred = denormalized(y_test_target_scaled, y_test_pred_scaled, scaler_target)
    test_mae_list, test_mse_list, test_rmse_list, test_mape_list = evaluate_metric(y_test_true, y_test_pred)

    if SINGLE_PERIOD: period_str = "single_period"
    else: period_str = "multi_period"
    

    # result_file_name = None

    # write to excel multi sheet
    df_train_true = pd.DataFrame(y_train_true.reshape(y_train_true.shape[0], y_train_true.shape[1]))
    df_train_pred = pd.DataFrame(y_train_pred.reshape(y_train_pred.shape[0], y_train_pred.shape[1]))

    df_test_true  = pd.DataFrame(y_test_true.reshape(y_test_true.shape[0], y_test_true.shape[1]))
    df_test_pred  = pd.DataFrame(y_test_pred.reshape(y_test_pred.shape[0], y_test_pred.shape[1]))

    evaluation = pd.DataFrame({"method_name": method_name,
                                "location_name": location_name,
                                "target_name": target_name,
                                "period_ahead": period_str,
                                "cv_name": cv_names[i],
                                "execution_time": execution_time,
                                "train_mae": np.mean(train_mae_list).round(3),
                                "train_mse": np.mean(train_mse_list).round(3),
                                "train_rmse": np.mean(train_rmse_list).round(3),
                                "train_mape": np.mean(train_mape_list).round(3),
                                "test_mae": np.mean(test_mae_list).round(3),
                                "test_mse": np.mean(test_mse_list).round(3),  
                                "test_rmse": np.mean(test_rmse_list).round(3),
                                "test_mape": np.mean(test_mape_list).round(3)
                                }, index=[i])

    with pd.ExcelWriter(result_file_path_cv + '.xlsx') as writer:
        evaluation.to_excel(writer, sheet_name='error evaluation')
        df_train_true.to_excel(writer, sheet_name='train_true')
        df_train_pred.to_excel(writer, sheet_name='train_pred')
        df_test_true.to_excel(writer, sheet_name='test_true')
        df_test_pred.to_excel(writer, sheet_name='test_pred')
        pd.DataFrame(train_mae_list).to_excel(writer, sheet_name='train_mae')
        pd.DataFrame(test_mae_list).to_excel(writer, sheet_name='test_mae')
        pd.DataFrame(train_mse_list).to_excel(writer, sheet_name='train_mse')
        pd.DataFrame(test_mse_list).to_excel(writer, sheet_name='test_mse')
        pd.DataFrame(train_rmse_list).to_excel(writer, sheet_name='train_rmse')
        pd.DataFrame(test_rmse_list).to_excel(writer, sheet_name='test_rmse')
        pd.DataFrame(train_mape_list).to_excel(writer, sheet_name='train_mape')
        pd.DataFrame(test_mape_list).to_excel(writer, sheet_name='test_mape')

    df_evaluations = pd.concat([df_evaluations, evaluation], axis=0)


    # plot result
    if SINGLE_PERIOD:
        plot_result(true=y_train_true, pred=y_train_pred, list=train_mae_list, 
                    input_width=INPUT_WIDTH, target_width=INPUT_WIDTH, single_period=SINGLE_PERIOD, 
                    title="Train MAE", path=result_file_path_cv)

        plot_result(true=y_test_true, pred=y_test_pred, list=test_mae_list, 
                    input_width=INPUT_WIDTH, target_width=INPUT_WIDTH, single_period=SINGLE_PERIOD, 
                    title="Test MAE", path=result_file_path_cv)
    else:
        plot_result(true=y_train_true, pred=y_train_pred, list=train_mae_list, 
                    input_width=INPUT_WIDTH, target_width=TARGET_WIDTH, single_period=SINGLE_PERIOD, 
                    title="Train MAE", path=result_file_path_cv)

        plot_result(true=y_test_true, pred=y_test_pred, list=test_mae_list, 
                    input_width=INPUT_WIDTH, target_width=TARGET_WIDTH, single_period=SINGLE_PERIOD, 
                    title="Test MAE", path=result_file_path_cv)

df_evaluations.to_excel(result_directory + "_" + method_name + "_" + location_name + "_" + \
                        target_name + "_target_width_" + str(TARGET_WIDTH) + "_evaluations.xlsx")