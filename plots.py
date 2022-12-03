import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.
from sklearn.metrics import confusion_matrix

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    # TODO: Make plots for loss curves and accuracy curves.
    # TODO: You do not have to return the plots.
    # TODO: You can save plots as files by codes here or an interactive way according to your preference.
    
    plt.rcParams["figure.figsize"] = (15, 5)    
    plt.subplot(1, 2, 1)

    # plot the loss curve...
    plt.title('Loss Curve')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)    
    # Plot the accuracy curve...
    plt.title('Accuracy Curve')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.legend()
       
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()

def plot_confusion_matrix(results, class_names):
    # TODO: Make a confusion matrix plot.
    # TODO: You do not have to return the plots.
    # TODO: You can save plots as files by codes here or an interactive way according to your preference.
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    #plt.rcParams.update(matplotlib.rcParamsDefault)
    
    yTrue = list(zip(*results))[0]
    yPred = list(zip(*results))[1]
    
    cm = confusion_matrix(yTrue, yPred)
    cmNorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cmNorm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap=plt.get_cmap('Blues'))
    plt.show()
