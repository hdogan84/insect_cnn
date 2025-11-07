import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image

import sys, os
from pathlib import Path
import json, random

sys.path.append(str(Path().resolve().parent))
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import average_precision_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from tools.plot_model_results import plot_train_val_acc_loss

seed_value = 42
os.environ["PYTHONHASHSEED"] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# Note: Use conda tf-gpu environment. 
project_root = Path().resolve().parent
config_dir = project_root / "config"
model_dir = project_root / "models"
spec_dir = "../data/image_data"

exp_no = 2

# -------------------------------
# 1. Load Json config and the Val Dataset
# -------------------------------
config_path = os.path.join(config_dir, f"exp_{exp_no}.json")

with open(config_path, "r") as f:
    config = json.load(f)


backbone_name = config.get("model_name")
model_name = f"{backbone_name}_exp{exp_no}.h5"  # your saved model file

batch_size = config.get("batch_size", 32)


val_data_generator = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.2,
                                   )


validation_data  = val_data_generator.flow_from_directory(directory = spec_dir,
                                                   target_size = (224, 224),
                                                   class_mode = 'binary',
                                                   subset = "validation",
                                                   batch_size = batch_size,
                                                   shuffle = False)


test_data_generator = ImageDataGenerator(rescale = 1./255)

test_data  = test_data_generator.flow_from_directory(directory = "../data/full_images_classified",
                                                   target_size = (224, 224),
                                                   class_mode = 'binary',
                                                   batch_size = batch_size,
                                                   shuffle = False)

# Map the indices
test_to_train_index = {test_idx: validation_data.class_indices[cls_name] 
                       for cls_name, test_idx in test_data.class_indices.items()}


test_data.classes = np.array([test_to_train_index[i] for i in test_data.classes])

subset_class_indices = {k: v for k, v in validation_data.class_indices.items() if k in test_data.class_indices}
test_data.class_indices = subset_class_indices

# -------------------------------
# 2. Load the saved model
# -------------------------------
model_path = model_dir / model_name
model = tf.keras.models.load_model(model_path)

# -------------------------------
# 3. Evaluate the model
# -------------------------------
#loss, acc = model.evaluate(test_data, verbose=1)
#print(f"Test Accuracy: {acc:.4f}")
#print(f"Test Loss: {loss:.4f}")

# -------------------------------
# 4. Predictions
# -------------------------------
pred_probs = model.predict(test_data)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = test_data.classes

unique_classes = sorted(set(int(i) for i in true_classes) | set(int(i) for i in pred_classes))
print("Unique class IDs:", unique_classes)

# 2️⃣ Create mapping old_index -> new_index (0..N-1)
mapping = {old: new for new, old in enumerate(unique_classes)}
print("Mapping:", mapping)

# 3️⃣ Map both arrays
true_classes_mapped = np.array([mapping[i] for i in true_classes])
pred_classes_mapped = np.array([mapping[i] for i in pred_classes])


# slice the class names using unique_classes
idx_to_class = {v: k for k, v in validation_data.class_indices.items()}
class_labels = [idx_to_class[i] for i in unique_classes if i in idx_to_class]

# -------------------------------
# 5. Confusion matrix
# -------------------------------
# Define save path for figures
save_path = project_root / "results" / "figures"
metrics_save_path = project_root / "results" / "metrics"

cm = np.round( confusion_matrix(true_classes_mapped, pred_classes_mapped, normalize = 'true'), 2 )
per_class_accuracy = np.diag(cm)
np.save(metrics_save_path / f"Exp{exp_no}_per_class_accuracy_test.npy", per_class_accuracy)

plt.figure(figsize=(36, 36))
sns.heatmap(cm, annot = True, cmap = plt.cm.Blues, xticklabels = class_labels, 
                yticklabels = class_labels, cbar= False, square = True, 
                annot_kws = {"size": 6}, linewidths = 1, linecolor = 'black')

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig(save_path / f"Exp{exp_no}_conf_mat_test.png", dpi=300, bbox_inches='tight')

# Close the figure to free memory (important in notebooks)
plt.close()

# -------------------------------
# 6. Optional: Classification report
# -------------------------------

report = classification_report(
    true_classes_mapped,
    pred_classes_mapped,
    target_names=class_labels
)

output_path = metrics_save_path / f"Exp{exp_no}_classification_report.txt"

# Write it to a file
with open(output_path, "w") as f:
    f.write("Classification Report\n")
    f.write("=====================\n\n")
    f.write(report)




