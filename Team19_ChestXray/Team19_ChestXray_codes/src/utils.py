import params
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

    
def plot_train_metrics(model_history, model_name, results_folder, run_timestamp):
    """
        Generate and save figure with plot of train and validation losses.
    """
    # extract data from history dict
    train_losses = model_history.history['loss']
    val_losses = model_history.history['val_loss']

    train_acc = model_history.history['binary_accuracy']
    val_acc = model_history.history['val_binary_accuracy']

    # define filenames
    loss_filename = f'{model_name}_loss_plot_{run_timestamp}.png'
    loss_fig_path = os.path.join(results_folder, loss_filename)
    acc_filename = f'{model_name}_accuracy_plot_{run_timestamp}.png'
    acc_fig_path = os.path.join(results_folder, acc_filename)

    # generate and save loss plot
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xticks(np.arange(0, len(train_losses), step=1))
    plt.xlabel('Epoch Number', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title(f'{model_name} Model', fontsize=18)
    plt.legend(('Training', 'Test'))
    plt.savefig(loss_fig_path)

    # clear axes and figure to reset for next plot
    plt.cla()
    plt.clf()

    # generate and save accuracy plot
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.xticks(np.arange(0, len(train_acc), step=1))
    plt.xlabel('Epoch Number', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.title(f'{model_name} Model', fontsize=18)
    plt.legend(('Training', 'Test'))
    plt.savefig(acc_fig_path)


def plot_ROC(disease_classes, test_Y, pred_Y, model_name):
    fig, c_ax = plt.subplots(1, 1, figsize=(10, 10))
    for (idx, c_label) in enumerate(disease_classes):
        fpr, tpr, thresholds = roc_curve(test_Y[:, idx].astype(int), pred_Y[:, idx])
        c_ax.plot(fpr, tpr, label=f'{c_label} (AUC = {auc(fpr, tpr):.3f})')

    c_ax.plot([0, 1], [0, 1], 'k--', lw=2)
    c_ax.legend(loc="lower right")
    c_ax.set_title(f'ROC Curve of {model_name}', fontsize=18)
    c_ax.set_xlabel('False Positive Rate', fontsize=18)
    c_ax.set_ylabel('True Positive Rate', fontsize=18)
    ROC_image_path = os.path.join(params.RESULTS_FOLDER, model_name, model_name + '_ROC.png')
    fig.savefig(ROC_image_path)


def save_model(model, model_name, results_folder, run_timestamp='unspecified'):
    """
        Generate and save figure with plot of train and validation losses.
    """
    # save model config
    json_filename = f'{model_name}_{run_timestamp}.json'
    output_json = os.path.join(results_folder, json_filename)
    with open(output_json, 'w') as json_file:
        json_file.write(model.to_json())

    # save trained model weights
    weights_filename = f'{model_name}_{run_timestamp}.hdf5'
    output_weights = os.path.join(results_folder, weights_filename)
    model.save_weights(output_weights)

    # Create symbolic link to the most recent weights (to use for testing)
    symlink_path = os.path.join(results_folder, f'{model_name}_weights_latest.hdf5')
    try:
        os.symlink(output_weights, symlink_path)
    except FileExistsError:
        # If the symlink already exist, delete and create again
        os.remove(symlink_path)
        os.symlink(output_weights, symlink_path)
    print(f'Created symbolic link to final weights -> {symlink_path}')
