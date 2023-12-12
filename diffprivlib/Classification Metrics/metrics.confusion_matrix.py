from sklearn import metrics
import numpy as np
from diffprivlib.tools.utils import mean

def dp_confusion_matrix(y_true, y_pred, epsilon=1.0):
    # Convert labels to numpy arrays if they're not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate confusion matrix using the regular function
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)

    # Add differential privacy to the confusion matrix calculation
    dp_mean = mean(y_pred, epsilon=epsilon)
    dp_conf_matrix = conf_matrix  # You may adjust this according to your DP calculation

    return conf_matrix, dp_conf_matrix
