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

# Sequence data
X_full = pickle.load(open(X_file, "rb"))
group = list(X_full['group'])


protein_seq = X_full['protein_sequence']
# protein_length = np.max([len(seq) for seq in list(protein_seq)])


#   Protein sequences
protein = X_full['one_hots']

# Expression level
y_full = pickle.load(open(y_file, "rb"))

group_y = list(y_full['group'])
assert all(group[i] == group_y[i] for i in range(len(group)))

y = y_full['Protein_'+selected_tissue].copy()
y = np.expand_dims(y, axis=2)




################################################################
# Protein --> Protein level
################################################################

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