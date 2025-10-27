
import matplotlib.pyplot as plt
from matplotlib import image


def plot_train_val_acc_loss(history, save_path, model_name, exp_no):

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

