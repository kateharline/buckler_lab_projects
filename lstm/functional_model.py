import platform
import os
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr as cor
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import keras.layers.advanced_activations as advanced_activations
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
from keras.initializers import he_normal
import keras.optimizers as optimizers
import keras.backend as backend
import keras.layers
import h5py
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

################################################################
# Script parameters
################################################################
# Working directory
if 'Ubuntu' in platform.platform():
    os.chdir('/home/gr226/Documents/protein_ml')
else:
    os.chdir('/workdir/gr226')

# Input files
X_file = "data/X.pkl"
y_file = "data/y.pkl"
DNA_encoding_file = "data/DNA_onehot.pkl"
protein_encoding_file = "data/protein_eigen.pkl"

# Selected tissue
selected_tissue = 'X6_7_internode'

# Learning parameters
batch_size = 256
n_epochs = 100
min_delta = 0 # Minimum difference for improvement on validation loss in early stopping
patience = 5 # Patience in early stopping

################################################################
# Functions
################################################################


def prediction_accuracy(y_true, y_pred):
    c12 = backend.sum((y_true - backend.mean(y_true)) * (y_pred - backend.mean(y_pred)))
    c11 = backend.sum(backend.square(y_true - backend.mean(y_true)))
    c22 = backend.sum(backend.square(y_pred - backend.mean(y_pred)))
    return c12/backend.sqrt(c11*c22)


# DNA sequence scan
def DNA_scan(input_sequence, cnn_layers=4, fcn_layers=1):

    x = Conv2D(64, kernel_size=(4, 6), padding='valid')(input_sequence)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(1,6), padding='same')(x)
    x = Dropout(0.25)(x)

    for _ in range(1, cnn_layers):
        x = Conv2D(64, kernel_size=(1,6), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1,6), padding='same')(x)
        x = Dropout(0.25)(x)

    x = Flatten()(x)

    for _ in range(fcn_layers):
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)

    return x

# Protein sequence scan
def protein_scan(input_sequence, cnn_layers=4, fcn_layers=1):
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

    return x


################################################################
# Model parameters
################################################################
# Model performance metric
model_metric = prediction_accuracy
metric_name = model_metric.__name__
plt_metric_name = metric_name.replace('_', ' ').capitalize()

# Save models every 10 epochs?
make_checkpoints = False

# Output
output_folder = 'data/functional_model'
os.system('mkdir ' + output_folder)

model_dir = os.path.join(output_folder, 'tmp')
os.system('mkdir ' + model_dir)

################################################################
# Input
################################################################
# Dictionaries
DNA_encoding = pickle.load(open(DNA_encoding_file, "rb"))
protein_encoding = pickle.load(open(protein_encoding_file, "rb"))

# Sequence data
X_full = pickle.load(open(X_file, "rb"))
group = list(X_full['group'])

DNA_seq = X_full['TSS_promoter']
DNA_length = np.max([len(seq) for seq in list(DNA_seq)])

protein_seq = X_full['protein_sequence']
protein_length = np.max([len(seq) for seq in list(protein_seq)])

#   Promoter sequences
promoter = np.zeros((DNA_seq.shape[0], 1500, len(DNA_encoding['A'])))
promoter_length = 1500

for i in range(DNA_seq.shape[0]):
    seq = DNA_seq[i]
    for j in range(promoter_length):
        promoter[i][j] = DNA_encoding[seq[j]]

promoter = np.expand_dims(promoter, axis=3)
promoter = np.transpose(promoter, (0, 2, 1, 3))

#   5'-UTR sequences
UTR = np.zeros((DNA_seq.shape[0], DNA_length-promoter_length, len(DNA_encoding['A'])))

for i in range(DNA_seq.shape[0]):
    seq = DNA_seq[i]
    for j in range(promoter_length, len(seq)):
        UTR[i][j-promoter_length] = DNA_encoding[seq[j]]

UTR = np.expand_dims(UTR, axis=3)
UTR = np.transpose(UTR, (0, 2, 1, 3))

#   Protein sequences
protein = np.zeros((protein_seq.shape[0], protein_length, len(protein_encoding['A'])))

for i in range(protein_seq.shape[0]):
    seq = protein_seq[i]
    for j in range(len(seq)):
        protein[i][j] = protein_encoding[seq[j]]

# Expression level
y_full = pickle.load(open(y_file, "rb"))

group_y = list(y_full['group'])
assert all(group[i] == group_y[i] for i in range(len(group)))

y = y_full['Protein_'+selected_tissue].copy()
y = np.expand_dims(y, axis=2)

# Training-Validation split
train_indices = np.where(np.array(group) == 'train')
val_indices = np.where(np.array(group) == 'val')
test_indices = np.where(np.array(group) == 'test')

#   Promoter sequences
promoter_train = promoter[train_indices]
promoter_val = promoter[val_indices]
promoter_test = promoter[test_indices]

#   Promoter sequences
UTR_train = UTR[train_indices]
UTR_val = UTR[val_indices]
UTR_test = UTR[test_indices]

#   protein sequences
protein_train = protein[train_indices]
protein_val = protein[val_indices]
protein_test = protein[test_indices]

#   Expression level
y_train = y[train_indices]
y_val = y[val_indices]
y_test = y[test_indices]

################################################################
# TSS promoter, 5'UTR, Protein --> Protein level
################################################################
model_name = 'TSS+UTR+P'

#   Input
promoter_sequence = Input(shape=promoter.shape[1:])
UTR_sequence = Input(shape=UTR.shape[1:])
protein_sequence = Input(shape=protein.shape[1:])

#   Motif scans
promoter_scan = DNA_scan
UTR_scan = DNA_scan

promoter_motif = promoter_scan(promoter_sequence)
UTR_motif = UTR_scan(UTR_sequence)
protein_motif = protein_scan(protein_sequence)

#   FC layers on concatenated representations
merged = keras.layers.concatenate([promoter_motif, UTR_motif, protein_motif], name='output')
expression = Dense(64, activation='relu')(merged)
expression = Dense(64, activation='relu')(expression)
expression = Dense(64, activation='relu')(expression)

#   Output
expression = Dense(1, activation='relu')(expression)

# Model
model = Model(inputs=[promoter_sequence, UTR_sequence, protein_sequence],
              outputs=[expression],
              name='protein_level')

# Inspection
model.summary()
print('Output shape: ' + str(model.output_shape))

# Compilation
model.compile(optimizer=optimizers.Adam(),
              loss='mse',
              metrics=[model_metric])

# Training
callbacks = [EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=0, mode='auto')]

if make_checkpoints:
    callbacks += ModelCheckpoint(filepath=os.path.join(model_dir, model_name + '__epoch={epoch:02d}.h5'), period=10)

fit = model.fit([promoter_train, UTR_train, protein_train], [y_train],
                validation_data=([promoter_val, UTR_val, protein_val], [y_val]),
                batch_size=batch_size, epochs=n_epochs, callbacks=callbacks)

# Saving fitted model
try:
    model.save(os.path.join(model_dir, model_name) + '.h5')
except ValueError:
    warnings.warn('Model could not be saved')

# Validation
ypred_train = model.predict([promoter_train, UTR_train, protein_train])
ypred_val = model.predict([promoter_val, UTR_val, protein_val])
ypred_test = model.predict([promoter_test, UTR_test, protein_test])

cor_train = cor(y_train, ypred_train)[0][0]
cor_val = cor(y_val, ypred_val)[0][0]
cor_test = cor(y_test, ypred_test)[0][0]

# Plot training history
accuracy_train = fit.history[list(fit.history)[-1]]
accuracy_val = fit.history[list(fit.history)[1]]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(accuracy_train, color='g', label='Training')
ax.plot(accuracy_val, color='b', label='Validation')
ax.set(title=model_name.replace('__', ', ').replace('_', ' '),
       xlabel='Epoch',
       ylabel=plt_metric_name)
ax.legend(loc='best')
fig.savefig(os.path.join(model_dir, model_name + '_history--' + selected_tissue + '.png'))

# Training and validation correlation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y_train, ypred_train, color='g',
           label='Training: r='+str(np.round(cor_train, 3)), alpha=0.15)
ax.scatter(y_val, ypred_val, color='b',
           label='Validation: r='+str(np.round(cor_val, 3)), alpha=0.15)
ax.set(title=model_name.replace('__', ', ').replace('_', ' '),
       xlabel='Observed',
       ylabel='Predicted')
ax.legend(loc='best')
fig.savefig(os.path.join(model_dir, model_name + '-predicting_validation_set--' + selected_tissue + '.png'))

# Training and validation correlation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y_test, ypred_test, color='r',
           label='Testing: r=' + str(np.round(cor_test, 3)), alpha=0.15)
ax.set(title=model_name.replace('__', ', ').replace('_', ' ')+' (Test)',
       xlabel='Observed',
       ylabel='Predicted')
ax.legend(loc='best')
fig.savefig(os.path.join(model_dir, model_name + '-predicting_test_set--' + selected_tissue + '.png'))

################################################################
# TSS promoter, 5'UTR --> Protein level
################################################################
model_name = 'TSS+UTR'

#   Input
promoter_sequence = Input(shape=promoter.shape[1:])
UTR_sequence = Input(shape=UTR.shape[1:])

#   Motif scans
promoter_scan = DNA_scan
UTR_scan = DNA_scan

promoter_motif = promoter_scan(promoter_sequence)
UTR_motif = UTR_scan(UTR_sequence)

#   FC layers on concatenated representations
merged = keras.layers.concatenate([promoter_motif, UTR_motif], name='output')
expression = Dense(64, activation='relu')(merged)
expression = Dense(64, activation='relu')(expression)
expression = Dense(64, activation='relu')(expression)

#   Output
expression = Dense(1, activation='relu')(expression)

# Model
model = Model(inputs=[promoter_sequence, UTR_sequence],
              outputs=[expression],
              name='protein_level')

# Inspection
model.summary()
print('Output shape: ' + str(model.output_shape))

# Compilation
model.compile(optimizer=optimizers.Adam(),
              loss='mse',
              metrics=[model_metric])

# Training
callbacks = [EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=0, mode='auto')]

if make_checkpoints:
    callbacks += ModelCheckpoint(filepath=os.path.join(model_dir, model_name + '__epoch={epoch:02d}.h5'), period=10)

fit = model.fit([promoter_train, UTR_train], [y_train],
                validation_data=([promoter_val, UTR_val], [y_val]),
                batch_size=batch_size, epochs=n_epochs, callbacks=callbacks)

# Saving fitted model
try:
    model.save(os.path.join(model_dir, model_name) + '.h5')
except ValueError:
    warnings.warn('Model could not be saved')

# Validation
ypred_train = model.predict([promoter_train, UTR_train])
ypred_val = model.predict([promoter_val, UTR_val])
ypred_test = model.predict([promoter_test, UTR_test])

cor_train = cor(y_train, ypred_train)[0][0]
cor_val = cor(y_val, ypred_val)[0][0]
cor_test = cor(y_test, ypred_test)[0][0]

# Plot training history
accuracy_train = fit.history[list(fit.history)[-1]]
accuracy_val = fit.history[list(fit.history)[1]]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(accuracy_train, color='g', label='Training')
ax.plot(accuracy_val, color='b', label='Validation')
ax.set(title=model_name.replace('__', ', ').replace('_', ' '),
       xlabel='Epoch',
       ylabel=plt_metric_name)
ax.legend(loc='best')
fig.savefig(os.path.join(model_dir, model_name + '_history--' + selected_tissue + '.png'))

# Training and validation correlation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y_train, ypred_train, color='g',
           label='Training: r='+str(np.round(cor_train, 3)), alpha=0.15)
ax.scatter(y_val, ypred_val, color='b',
           label='Validation: r='+str(np.round(cor_val, 3)), alpha=0.15)
ax.set(title=model_name.replace('__', ', ').replace('_', ' '),
       xlabel='Observed',
       ylabel='Predicted')
ax.legend(loc='best')
fig.savefig(os.path.join(model_dir, model_name + '-predicting_validation_set--' + selected_tissue + '.png'))

# Training and validation correlation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y_test, ypred_test, color='r',
           label='Testing: r=' + str(np.round(cor_test, 3)), alpha=0.15)
ax.set(title=model_name.replace('__', ', ').replace('_', ' ')+' (Test)',
       xlabel='Observed',
       ylabel='Predicted')
ax.legend(loc='best')
fig.savefig(os.path.join(model_dir, model_name + '-predicting_test_set--' + selected_tissue + '.png'))

################################################################
# Protein --> Protein level
################################################################
model_name = 'P'

#   Input
protein_sequence = Input(shape=protein.shape[1:])

#   Motif scans
protein_motif = protein_scan(protein_sequence)

#   FC layers on concatenated representations
expression = Dense(64, activation='relu')(protein_motif)
expression = Dense(64, activation='relu')(expression)

#   Output
expression = Dense(1, activation='relu')(expression)

# Model
model = Model(inputs=[protein_sequence],
              outputs=[expression],
              name='protein_level')

# Inspection
model.summary()
print('Output shape: ' + str(model.output_shape))

# Compilation
model.compile(optimizer=optimizers.Adam(),
              loss='mse',
              metrics=[model_metric])

# Training
callbacks = [EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=0, mode='auto')]

if make_checkpoints:
    callbacks += ModelCheckpoint(filepath=os.path.join(model_dir, model_name + '__epoch={epoch:02d}.h5'), period=10)

fit = model.fit([protein_train], [y_train],
                validation_data=([protein_val], [y_val]),
                batch_size=batch_size, epochs=n_epochs, callbacks=callbacks)

# Saving fitted model
try:
    model.save(os.path.join(model_dir, model_name) + '.h5')
except ValueError:
    warnings.warn('Model could not be saved')

# Validation
ypred_train = model.predict([protein_train])
ypred_val = model.predict([protein_val])
ypred_test = model.predict([protein_test])

cor_train = cor(y_train, ypred_train)[0][0]
cor_val = cor(y_val, ypred_val)[0][0]
cor_test = cor(y_test, ypred_test)[0][0]

# Plot training history
accuracy_train = fit.history[list(fit.history)[-1]]
accuracy_val = fit.history[list(fit.history)[1]]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(accuracy_train, color='g', label='Training')
ax.plot(accuracy_val, color='b', label='Validation')
ax.set(title=model_name.replace('__', ', ').replace('_', ' '),
       xlabel='Epoch',
       ylabel=plt_metric_name)
ax.legend(loc='best')
fig.savefig(os.path.join(model_dir, model_name + '_history--' + selected_tissue + '.png'))

# Training and validation correlation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y_train, ypred_train, color='g',
           label='Training: r='+str(np.round(cor_train, 3)), alpha=0.15)
ax.scatter(y_val, ypred_val, color='b',
           label='Validation: r='+str(np.round(cor_val, 3)), alpha=0.15)
ax.set(title=model_name.replace('__', ', ').replace('_', ' '),
       xlabel='Observed',
       ylabel='Predicted')
ax.legend(loc='best')
fig.savefig(os.path.join(model_dir, model_name + '-predicting_validation_set--' + selected_tissue + '.png'))

# Training and validation correlation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y_test, ypred_test, color='r',
           label='Testing: r=' + str(np.round(cor_test, 3)), alpha=0.15)
ax.set(title=model_name.replace('__', ', ').replace('_', ' ')+' (Test)',
       xlabel='Observed',
       ylabel='Predicted')
ax.legend(loc='best')
fig.savefig(os.path.join(model_dir, model_name + '-predicting_test_set--' + selected_tissue + '.png'))