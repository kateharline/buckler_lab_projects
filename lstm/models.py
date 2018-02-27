
def lstm_class(input_sequence, lstm_layers=4, units=32, fcn_layers=2):
    '''
    use the functional API to instantiate layers in LSTM
    :param input_sequence: np multi-dim array of encoded protein sequences
    :param lstm_layers: int number of lstm layers
    :param fcn_layers: int number of fully connected layers
    :return: model with layers applied
    '''
    seq_return = False

    if lstm_layers > 1:
        seq_return = True

    x = LSTM(units, return_sequences=seq_return)(input_sequence)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.5)(x)

    for i in range(1, lstm_layers):
        if i < lstm_layers - 1:
            seq_return = True
        else:
            seq_return = False

        x = LSTM(units, return_sequences=seq_return)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.5)(x)

    for _ in range(fcn_layers):
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.5)(x)

    x = Dense(1, activation='sigmoid')(x)

    return x

def lstm_cont(input_sequence, lstm_layers=4, units=32, fcn_layers=2):
    '''
    use the functional API to instantiate layers in LSTM
    :param input_sequence: np multi-dim array of encoded protein sequences
    :param lstm_layers: int number of lstm layers
    :param fcn_layers: int number of fully connected layers
    :return: model with layers applied
    '''
    seq_return = False

    if lstm_layers > 1:
        seq_return = True

    x = LSTM(units, return_sequences=seq_return)(input_sequence)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.5)(x)

    for i in range(1, lstm_layers):
        if i < lstm_layers - 1:
            seq_return = True
        else:
            seq_return = False

        x = LSTM(units, return_sequences=seq_return)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.5)(x)

    for _ in range(fcn_layers):
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.5)(x)

    x = Dense(1, activation='relu')(x)

    return x

def make_lstm_cont(protein_i, max_length, seq_type, num_layers, u_per_layer):
    '''
    instantiate keras model
    :param X: np array of x values
    :param max_length: int max length you want to embed
    :param sequence type: string na (mucleic acid) or protein
    :return: keras model object
    '''
    # switch
    oh_lengths = {'protein': 21, 'na': 5}
    oh_length = oh_lengths[seq_type]
    metrics = [y_pred_mean, prediction_accuracy]  # 'accuracy'
    protein = Input(shape=protein_i.shape[1:])
    # instantiate the model
    fc = lstm_cont(protein, lstm_layers=num_layers, units=u_per_layer)

    model = Model(inputs=[protein],
                  outputs=[fc],
                  name='protein_level')

    # Inspection
    model.summary()
    print('Output shape: ' + str(model.output_shape))

    # Compilation
    model.compile(optimizer=optimizers.Adam(),
                  loss='mse',
                  metrics=metrics)

    return model

def make_lstm_class(protein_i, max_length, seq_type, learn_rate=0.001):
    '''
    instantiate keras model
    :param X: np array of x values
    :param max_length: int max length you want to embed
    :param sequence type: string na (mucleic acid) or protein
    :return: keras model object
    '''
    # switch
    oh_lengths = {'protein': 21, 'na': 5}
    oh_length = oh_lengths[seq_type]
    metrics = [y_pred_mean, 'accuracy']  # 'accuracy'
    protein = Input(shape=protein_i.shape[1:])
    # instantiate the model
    fc = lstm_class(protein)
    fc = fc_apply(fc)

    model = Model(inputs=[protein],
                  outputs=[fc],
                  name='protein_level')

    # Inspection
    model.summary()
    print('Output shape: ' + str(model.output_shape))

    # Compilation
    model.compile(optimizer=optimizers.Adam(lr=learn_rate),
                  loss='binary_crossentropy',
                  metrics=metrics)

    return model

# Protein sequence scan
def cnn_class(input_sequence, cnn_layers=4, fcn_layers=1):
    '''
    use the functional API to instantiate layers in CNN
    :param input_sequence: np multi-dim array of encoded protein sequences
    :param cnn_layers: int number of convolutional layers
    :param fcn_layers: int number of fully connected layers
    :return: model with layers applied
    '''

    x = Conv1D(64, kernel_size=5, padding='valid')(input_sequence)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(padding='same')(x)
    x = Dropout(0.25)(x)

    for _ in range(1, cnn_layers):
        x = Conv1D(64, kernel_size=5, padding='valid')(x)
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

    expression = Dense(1, activation='sigmoid')(expression)

    return x

# Protein sequence scan
def cnn_cont(input_sequence, cnn_layers=4, fcn_layers=1):
    '''
    use the functional API to instantiate layers in CNN
    :param input_sequence: np multi-dim array of encoded protein sequences
    :param cnn_layers: int number of convolutional layers
    :param fcn_layers: int number of fully connected layers
    :return: model with layers applied
    '''

    x = Conv1D(64, kernel_size=5, padding='valid')(input_sequence)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(padding='same')(x)
    x = Dropout(0.25)(x)

    for _ in range(1, cnn_layers):
        x = Conv1D(64, kernel_size=5, padding='valid')(x)
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

    x = Dense(1, activation='relu')(x)

    return x




def make_cnn_class(protein_i, max_length, seq_type):
    '''
    instantiate keras model
    :param X: np array of x values
    :param max_length: int max length you want to embed
    :param sequence type: string na (mucleic acid) or protein
    :return: keras model object
    '''
    # switch
    oh_lengths = {'protein':21, 'na':5}
    oh_length = oh_lengths[seq_type]
    metrics = [prediction_accuracy] # 'accuracy'
    protein = Input(shape=protein_i.shape[1:])
    # instantiate the model
    conv_protein = cnn_class(protein)

    model = Model(inputs=[protein],
              outputs=[conv_protein],
              name='protein_level')

    # Inspection
    model.summary()
    print('Output shape: ' + str(model.output_shape))

    # Compilation
    model.compile(optimizer=optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=metrics)

    return model


def make_cnn_cont(protein_i, max_length, seq_type):
    '''
    instantiate keras model
    :param X: np array of x values
    :param max_length: int max length you want to embed
    :param sequence type: string na (mucleic acid) or protein
    :return: keras model object
    '''
    # switch
    oh_lengths = {'protein':21, 'na':5}
    oh_length = oh_lengths[seq_type]
    metrics = [prediction_accuracy] # 'accuracy'
    protein = Input(shape=protein_i.shape[1:])
    # instantiate the model
    conv_protein = cnn_const(protein)

    model = Model(inputs=[protein],
              outputs=[conv_protein],
              name='protein_level')

    # Inspection
    model.summary()
    print('Output shape: ' + str(model.output_shape))

    # Compilation
    model.compile(optimizer=optimizers.Adam(),
                  loss='mse',
                  metrics=metrics)

    return model