from xlrd import open_workbook
from numpy import array, concatenate, sqrt, arange
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Activation, Input, TimeDistributed
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


def get_problem1_model():
    model = Sequential()
    model.add(LSTM(input_shape=(461, 1), return_sequences=True, units=461))
    model.add(Dropout(0.2))
    model.add(LSTM(461, return_sequences=True))
    model.add(TimeDistributed(Dense(461)))
    model.compile(loss='mse', optimizer='rmsprop')
    print(model.summary())
    plot_model(model, to_file='./model_1.png', show_shapes=True)
    return model


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


if __name__ == "__main__":
    fname = './washed data.xlsx'

    neurons = 461
    activation_function = 'tanh'
    loss = 'mse'
    optimizer = "adam"
    dropout = 0.25
    batch_size = 12
    epochs = 53

    first_col, col_list = get_excel_data_as_col(fname)
    data_list = make_continuous_data(col_list, 365)
    data = DataFrame(data_list, columns=first_col)
    # normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    n_train_time = 6 * 365
    print(reframed)
    data = reframed.values
    train = data[:n_train_time, :]
    test = data[n_train_time:, :]

    train_x, train_y = train[:, :-461], train[:, -461:]
    test_x, test_y = test[:, :-461], test[:, -461:]

    train_x = train_x.reshape(train_x.shape[1], train_x.shape[0], 1)
    test_x = test_x.reshape(test_x.shape[1], test_x.shape[0], 1)
    print(train_x.shape, train_y.shape)

    model = get_problem1_model()
    # #
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_x, test_y), verbose=2, shuffle=False)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history["val_loss"], label='test')
    pyplot.legend()
    pyplot.show()




