
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks, Sequential
from tensorflow.keras.optimizers import Adam

print(tf.__version__)

from sklearn.model_selection import train_test_split


train_data_generator = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.2,
                                   )

training_data  = train_data_generator.flow_from_directory(directory = 'data/train_data/spec',
                                                   #target_size = (224, 224),
                                                   class_mode = 'binary',
                                                   subset = "training", 
                                                   batch_size = 32)

validation_data  = train_data_generator.flow_from_directory(directory = 'data/train_data/spec',
                                                   #target_size = (224, 224),
                                                   class_mode = 'binary',
                                                   subset = "validation", 
                                                   batch_size = 32)


early_stopping = callbacks.EarlyStopping(monitor = 'val_loss',
                                         patience = 4,
                                         mode = 'min',
                                         restore_best_weights = True)

lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                        patience=3,
                                         factor=0.5,
                                         verbose=2,
                                         mode='min',
                                         min_lr = 1e-10)


inp_shape=(256,256,3)


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), 2, activation='relu',input_shape=inp_shape),
  tf.keras.layers.AvgPool2D(pool_size=3, strides=2, padding='valid', data_format=None),
  tf.keras.layers.Conv2D(64, (3,3), 2, activation='relu'),
  tf.keras.layers.AvgPool2D(pool_size=3, strides=2, padding='valid', data_format=None),
  tf.keras.layers.Conv2D(16, (3,3), 2, activation='relu'),
  tf.keras.layers.AvgPool2D(pool_size=3, strides=2, padding='valid', data_format=None),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1800, activation='relu'),
  tf.keras.layers.Dense(1264, activation='tanh'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])


### Example size check
#x = np.random.rand(1,256, 256, 3) 
#print(model(x).shape)

optimizer = Adam(learning_rate=0.0001) 

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(training_data, validation_data=validation_data, epochs=20, callbacks=[early_stopping,lr_plateau])


#from sklearn.metrics import confusion_matrix
#confusion_mtx = confusion_matrix(y_test, y_pred) 
#print(confusion_mtx)

model.save('my_model_epochs20.h5')

