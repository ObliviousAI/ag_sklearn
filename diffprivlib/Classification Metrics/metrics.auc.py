from sklearn import metrics
import numpy as np
from diffprivlib.tools.histograms import histogram

def dp_auc(y_true, y_score, epsilon=1.0, bins=10, range=None):
    # Convert labels to numpy arrays if they're not already
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Calculate AUC using the regular auc function
    auc = metrics.auc(y_true, y_score)

    # Add differential privacy to the AUC calculation
    dp_hist, bin_edges = histogram(y_score, epsilon=epsilon, bins=bins, range=range)
    dp_auc = dp_hist.sum() / len(y_true)

    return auc, dp_auc

# Example usage:
# auc, dp_auc = dp_auc(y_true, y_score, epsilon=1.0, bins=10, range=(0, 1))