import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image

import sys
from pathlib import Path

sys.path.append(str(Path().resolve().parent))
#sys.path.append(str(Path(__file__).resolve().parent.parent))

from os import listdir
from os.path import isfile, join

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, GlobalAveragePooling2D
from keras.applications import MobileNet, EfficientNetV2B0, ResNet50, ResNet50V2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from tools.metrics import AveragePrecisionCallback


# Note: Use conda tf-gpu environment. 


spec_dir = '../data/image_data/'


num_classes = 129
batch_size = 32
exp_no = 13
num_epochs = 50
model_name = "EfficientNetV2B0"



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

test_data_generator  = ImageDataGenerator(rescale = 1./255)
data_generator  = ImageDataGenerator(rescale = 1./255)



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

#test_data  = test_data_generator.flow_from_directory(directory = spec_dir,
#                                                   target_size = (224, 224),
#                                                   class_mode = 'binary',
#                                                   batch_size = batch_size)



# Model MobileNet
if model_name == "MobileNet":
    base_model = MobileNet(weights='imagenet', include_top=False)
elif model_name == "ResNet50":
    base_model = ResNet50(weights='imagenet', include_top=False)
elif model_name == "EfficientNetV2B0":
    base_model = EfficientNetV2B0(weights='imagenet', include_top=False)

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




# Go up one level to reach project root (same as src/parent)
project_root = Path().resolve().parent

# Define save path
save_path = project_root / "results" / "figures"
save_path.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss by epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='right')


plt.subplot(122)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model acc by epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='right')

# Save it to file (no display)
plt.tight_layout()  # optional: improves spacing
plt.savefig(save_path / f"{model_name}_exp{exp_no}_loss_acc.png", dpi=300, bbox_inches='tight')

# Close the figure to free memory (important in notebooks)
plt.close()


model.save(f'../models/{model_name}_127_species_exp{exp_no}.h5')





