
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import h5py

print(tf.__version__)

from sklearn.model_selection import train_test_split


# train = pd.read_csv('train.csv',header=None,index_col=None)
# train.head()


# np.save('train.npy', train)
train=np.load('train.npy')
# print(train.shape)

y_data=train[:,0]
# print(y_data.shape)

x_data=train[:,1:]

x_data=x_data.reshape([462,257,395,1])

# do train test split 
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=22)

# x_train.shape
# y_train.shape

inp_shape=(257,395,1)

# yy=tf.keras.layers.Conv2D(32,(3,3),2,input_shape=inp_shape)(x_train)
# yy.shape
# zz=tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', data_format=None)(yy)
# zz.shape
# tt=tf.keras.layers.Conv2D(8, (3,3), 2)(zz)
# tt.shape
# kk=tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', data_format=None)(tt)
# kk.shape
# tf.keras.layers.Flatten()(kk).shape


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), 2,activation='relu',input_shape=inp_shape),
  tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', data_format=None),
  tf.keras.layers.Conv2D(8, (3,3), 2,activation='relu'),
  tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', data_format=None),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1028, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


num_epochs=40
for ii in range(0,num_epochs):
    model.fit(x_train, y_train, epochs=1)
    model.evaluate(x_test,  y_test, verbose=2)


model.save('C:/Users/dgnhk_000/Downloads/ARSU 2017/20170330_Uhu/Waldschnepfe_recog/my_model_epochs36_v2.h5')

y_predict=model.predict(x_test)


y_pred = np.argmax(y_predict, axis=1)
y_pred


from sklearn.metrics import confusion_matrix
confusion_mtx = confusion_matrix(y_test, y_pred) 
print(confusion_mtx)


model.save('C:/Users/dgnhk_000/Downloads/ARSU 2017/20170330_Uhu/Waldschnepfe_recog/my_model_epochs20.h5')

