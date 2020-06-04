#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
"""

from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import seaborn as sns
sns.set_style("darkgrid")

import numpy as np

from tensorflow.keras import callbacks


class PlotTraining(callbacks.Callback):
    """
    TODO: Write Comment
    """

    def __init__(self, filepath=""):
        """
        TODO: Write Comment
        """

        super(PlotTraining, self).__init__()

        self.filepath = filepath
        self.reset()

    
    def reset(self):
        """
        TODO: Write Comment
        """

        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []

    
    def on_epoch_end(self, epoch, logs={}):
        """
        TODO: Write Comment
        """

        self.x.append(self.i+1)
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.i += 1
                        
        fig        = plt.figure(1, figsize=(16,9),dpi=300)
        (ax1, ax2) = fig.subplots(1,2)
        
        ax1.plot(self.x, self.losses,     label="Training Loss")
        ax1.plot(self.x, self.val_losses, label="Validation Loss")
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc='best')
        
        ax2.plot(self.x, self.acc,     label="Train Accuracy")
        ax2.plot(self.x, self.val_acc, label="Validation Accuracy")
        ax2.set_title('Model Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc='best')
        
        fig.tight_layout()
        fig.savefig(f"{self.filepath}ModelTraining.png", bbox_inches="tight", dpi=300)
        fig.clear()

def plot_training_history(histories, filepath):

    fig        = plt.figure(1, figsize=(16,8),dpi=300)
    (ax1, ax2) = fig.subplots(1,2)

    cmap  = plt.cm.get_cmap('tab20', 20)
    cm    = plt.cm.ScalarMappable(cmap=cmap)
    cm._A = []

    i = 0

    for history in histories:

        try:
            x = range(1, len(history[f"accuracy"]) + 1)
            prefix = ''
        except:
            prefix = 'output_'
            x = range(1, len(history[f"{prefix}accuracy"]) + 1)
        
        ax1.plot(x, history[f"{prefix}loss"],     label="Training Loss",   linestyle=':', alpha=0.8, color=cmap(i+1))
        ax1.plot(x, history[f"val_{prefix}loss"], label="Validation Loss", linestyle='-', alpha=0.8, color=cmap(i))
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        
        ax2.plot(x, history[f"{prefix}accuracy"],     label="Train Accuracy",      linestyle=':', alpha=0.8, color=cmap(i+1))
        ax2.plot(x, history[f"val_{prefix}accuracy"], label="Validation Accuracy", linestyle='-', alpha=0.8, color=cmap(i))
        ax2.set_title('Model Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        
        i += 2
    
    fig.tight_layout()
    fig.savefig(f"{filepath}ModelTraining.png", bbox_inches="tight", dpi=300)
    fig.clear()

def plot_image(text, index, adversarial_image, original_image, label_true, label_pred, limit, l2):
    """
    TODO: Write Comment
    """

    def plot(index, image, label, label_type=""):
        """
        TODO: Write Comment
        """

        if image.ndim == 4 and image.shape[0] == 1: image = image[0]

        plt.subplot(1,2,index)
        if image.shape[-1] == 3:
            plt.imshow(image.astype(np.uint8))
        else:
            plt.imshow(image[:,:,0].astype(np.uint8), cmap='gray')
        plt.xlabel(f"{label_type}:{label}")
        plt.xticks([]); plt.yticks([])

    plt.grid()
    
    plot(1, original_image,    label_true, "True")
    plot(2, adversarial_image, label_pred, "Predicted")

    plt.savefig(f"{text}/Index {index} True {label_true} Predicted {label_pred} with limit {limit} and L2 score of {l2:.4f}.png")