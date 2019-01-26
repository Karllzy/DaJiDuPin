from xlrd import open_workbook
from numpy import array, concatenate, sqrt, arange
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Input
from keras.optimizers import Adam
from keras.utils import plot_model
from matplotlib import pyplot


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


def get_model_1(lr, neurons, dropout):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    model.compile(loss='mae', optimizer=Adam(lr=lr))
    print(model.summary())
    plot_model(model, to_file='./model_1.png')
    return model


def train_model_1(model, train_x, train_y, test_x, test_y, epoches, batch_size):
    history = model.fit(train_x, train_y, epochs=epoches, batch_size=batch_size,
                        validation_data=(test_x, test_y), verbose=2, shuffle=False)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history["val_loss"], label='test')
    pyplot.legend()
    pyplot.show()
    return model


def build_model_1(inputs, output_size, neurons, activ_func, dropout, loss, optimizer):
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2]), activation=activ_func))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, return_sequences=True, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    model.summary()
    return model


def make_model_1():
    main_input = Input(shape=(461, ), name='main_input')
    lstm_1 = LSTM(461, activation='tanh',)(main_input)


def get_one_seq(i, reframed):
    range1 = arange(461, i+461, 1)
    range2 = arange(i+462, 922, 1)
    my_range = concatenate([range1, range2])
    reframed.drop(reframed.columns[my_range], axis=1, inplace=True)
    return reframed


if __name__ == "__main__":
    # load and generate data
    neurons = 461
    n_train_time = 6 * 365
    i = 0
    fname = '/home/zhenye/MeiSai/python_code_file/washed data.xlsx'
    first_col, col_list = get_excel_data_as_col(fname)
    data_list = make_continuous_data(col_list, 365)
    data = DataFrame(data_list, columns=first_col)
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    data = get_one_seq(i, reframed)
    print(data.head())
    data = data.values
    train = data[:n_train_time, :]
    test = data[n_train_time:, :]

    # train = []
    # test = []
    # for i in range(int(data.shape[0])):
    #     if i / 2 == 0:
    #      train.append(data[i, :])
    #     else:
    #      test.append(data[i, :])
    # test = array(test)
    # train = array(train)

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timeSteps, features]
    train_X = train_X.reshape((train_X.shape[0], 461, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 461, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(461, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.22))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=461, validation_data=(test_X, test_y), verbose=2,
                       shuffle=False)
    # history = model.fit(train_X, train_y, epochs=5000, batch_size=461, verbose=2,
    #                   shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

