from keras.layers import Activation, MaxPooling1D, Flatten, BatchNormalization, Input, LSTM, Dense, Dropout, Conv1D
from keras.layers.advanced_activations import LeakyReLU



def lstm(input_sequence, final_activation, lstm_layers=4, units=32, fcn_layers=2):

    seq_return = lstm_layers > 1

    x = LSTM(units, return_sequences=seq_return)(input_sequence)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.5)(x)

    for i in range(1, lstm_layers):
        seq_return = i < lstm_layers - 1

        x = LSTM(units, return_sequences=seq_return)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.5)(x)

    for _ in range(fcn_layers):
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.5)(x)

    x = Dense(1, activation=final_activation)(x)

    return x


def cnn(input_sequence, final_activation, layers=4, units=32, fcn_layers=2):


    x = Conv1D(units, kernel_size=5, padding='valid')(input_sequence)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(padding='same')(x)
    x = Dropout(0.25)(x)

    for _ in range(1, layers):
        x = Conv1D(units, kernel_size=5, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(padding='same')(x)
        x = Dropout(0.25)(x)

    x = Flatten()(x)

    for _ in range(fcn_layers):
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)

    x = Dense(1, activation=final_activation)(x)

    return x




