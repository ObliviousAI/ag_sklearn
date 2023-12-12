from sklearn import metrics
import numpy as np
from diffprivlib.tools.utils import mean

def dp_average_precision_score(y_true, y_score, epsilon=1.0):
    # Convert labels to numpy arrays if they're not already
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Calculate average precision using the regular function
    avg_precision = metrics.average_precision_score(y_true, y_score)

    # Add differential privacy to the average precision calculation
    dp_mean = mean(y_score, epsilon=epsilon)
    dp_avg_precision = dp_mean  # You may adjust this according to your DP calculation

    return avg_precision, dp_avg_precision
