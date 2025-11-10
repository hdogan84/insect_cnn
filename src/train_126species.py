import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image

import sys, os
from pathlib import Path
import json, random

sys.path.append(str(Path().resolve().parent))
#sys.path.append(str(Path(__file__).resolve().parent.parent))

from os import listdir
from os.path import isfile, join

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, GlobalAveragePooling2D
from keras.applications import MobileNet, MobileNetV2, ResNet50, ResNet50V2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from tools.metrics import AveragePrecisionCallback
from tools.plot_model_results import plot_train_val_acc_loss

seed_value = 42
os.environ["PYTHONHASHSEED"] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# Note: Use conda tf-gpu environment. 
project_root = Path().resolve().parent
config_dir = project_root / "config"
config_dir.mkdir(parents=True, exist_ok=True)
spec_dir = '../data/image_data/'



batch_size = 64
exp_no = 4
num_epochs = 60
model_name = "MobileNetV2"


train_data_generator = ImageDataGenerator(rescale = 1./255,
                                    validation_split = 0.2,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    brightness_range=(0.8, 1.2),
                                    fill_mode='nearest'
                                   )

val_data_generator = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.2,
                                   )



training_data  = train_data_generator.flow_from_directory(directory = spec_dir,
                                                   target_size = (224, 224),
                                                   class_mode = 'binary',
                                                   subset = "training",
                                                   batch_size = batch_size)

validation_data  = val_data_generator.flow_from_directory(directory = spec_dir,
                                                   target_size = (224, 224),
                                                   class_mode = 'binary',
                                                   subset = "validation",
                                                   batch_size = batch_size)


num_classes = len(training_data.class_indices)

# Save hyperparameters to json
params = {
    "num_classes": num_classes,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "model_name": model_name
}

# Save JSON file named after the experiment
json_path = config_dir / f"exp_{exp_no}.json"
with open(json_path, "w") as f:
    json.dump(params, f, indent=4)

# Model MobileNet
if model_name == "MobileNet":
    base_model = MobileNet(weights='imagenet', include_top=False)
elif model_name == "ResNet50":
    base_model = ResNet50(weights='imagenet', include_top=False)
elif model_name == "ResNet50V2":
    base_model = ResNet50V2(weights='imagenet', include_top=False)
elif model_name == "MobileNetV2":
    base_model = MobileNetV2(weights='imagenet', include_top=False)

# Freezer les couches
#for layer in base_model.layers:
#    layer.trainable = False


"""
# Unfreeze the last N trainable convolutional blocks
trainable = False
for layer in reversed(base_model.layers):
    if 'conv' in layer.name:
        if N == 0:
            break
        N -= 1
        trainable = True
    layer.trainable = trainable
"""

model = Sequential()
model.add(base_model) 
model.add(GlobalAveragePooling2D())
#model.add(Dropout(rate=0.3))
#model.add(Dense(units=1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# Callbacks
early_stopping = callbacks.EarlyStopping(monitor = 'val_accuracy',
                                         patience = 10,
                                         mode = 'max',
                                         restore_best_weights = True)

lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'val_accuracy',
                                        patience=4,
                                         factor=0.5,
                                         verbose=2,
                                         mode='max',
                                         min_lr = 1e-10)


ap_callback = AveragePrecisionCallback(validation_data, num_classes)

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',metrics=['accuracy'])


history = model.fit(training_data, validation_data=validation_data, epochs=num_epochs,
                    callbacks=[ap_callback, lr_plateau, early_stopping])




# Define save path for figures
save_path = project_root / "results" / "figures"
save_path.mkdir(parents=True, exist_ok=True)
plot_train_val_acc_loss(history, save_path, model_name, exp_no)


model.save(f'../models/{model_name}_exp{exp_no}.h5')





