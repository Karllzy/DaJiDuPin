from xlrd import open_workbook
from numpy import array, concatenate, sqrt, arange, random
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Activation, Input
from keras.optimizers import Adam
from keras.utils import plot_model
from matplotlib import pyplot
import numpy as np

def get_excel_data_as_col(fname):
    filename = open_workbook(fname)
    sheets = filename.nsheets

    sheet = filename.sheet_by_index(0)

    nrows = sheet.nrows
    ncols = sheet.ncols

    first_col = sheet.col_values(0)
    first_col.pop(0)
    col_list = []
    for i in range(1, ncols):
        col_data = sheet.col_values(i)
        col_data.pop(0)
        for i in range(len(col_data)):
            if col_data[i] == "":
                col_data[i] = 0.0
        col_list.append(col_data)
    return first_col, col_list


def make_continuous_data(spread_data, interval_number):
    data_list = []
    for i in range(len(spread_data)-1):
        array1 = array(spread_data[i])
        array2 = array(spread_data[i+1])
        for _ in range(interval_number):
            new_array = (array2 - array1) * _ / interval_number + array1
            data_list.append(new_array)
    return array(data_list, dtype=float)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('county%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('county%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('county%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_model_1():
    inputs = Input(shape=(461, ))
    x = Dense(461, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(461, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(461, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(461, activation='relu')(x)
    model = Model(inputs=inputs, output=x)
    model.compile(optimizer=Adam(lr=1e-5), loss='mse')
    return model


if __name__ == "__main__":
    fname = '/home/zhenye/MeiSai/python_code_file/washed data.xlsx'
    train_or_predetic = 1

    first_col, col_list = get_excel_data_as_col(fname)
    data_list = make_continuous_data(col_list, 365)
    data = DataFrame(data_list, columns=first_col)

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    n_train_time = 3 * 365
    data = reframed.values
    train = data[:n_train_time, :]
    test = data[n_train_time:, :]
    # print(data.shape)
    train = []
    test = []
    for i in range(int(data.shape[0])):
        if i/5 == 0:
         train.append([data[i, :]])
        else:
         test.append([data[i, :]])
    test = array(test)
    train = array(train)
    print(test.shape)
    print(train.shape)

    if 0:
        train_x, train_y = train[:, :-461], train[:, -461:]
        test_x, test_y = test[:, :-461], test[:, -461:]
        print(train_x.shape, train_y.shape)


        model = get_model_1()
        if train_or_predetic:
            print(train_x.shape, train_y.shape)
            history = model.fit(train_x, train_y, epochs=1000, batch_size=461, validation_data=(test_x, test_y))
            model.save_weights(filepath='./model1_weight.h5')
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history["val_loss"], label='test')
            pyplot.legend()
            pyplot.show()
        else:
            model.load_weights('./model1_weight.h5')
            # make a prediction



