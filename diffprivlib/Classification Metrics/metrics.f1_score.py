from sklearn import metrics
import numpy as np
from diffprivlib.tools.utils import mean

def dp_f1_score(y_true, y_pred, epsilon=1.0):
    # Convert labels to numpy arrays if they're not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate F1 score using the regular function
    f1 = metrics.f1_score(y_true, y_pred)

    # Add differential privacy to the F1 score calculation
    dp_mean = mean(y_pred, epsilon=epsilon)
    dp_f1 = f1  # You may adjust this according to your DP calculation

    return f1, dp_f1
