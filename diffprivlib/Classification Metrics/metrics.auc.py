from sklearn import metrics
import numpy as np
from diffprivlib.tools.histograms import histogram
from diffprivlib.accountant import BudgetAccountant


def auc(y_true, y_score, epsilon=1.0, bins=10, range=None):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    dp_hist, bin_edges = histogram(y_score, epsilon=epsilon, bins=bins, range=range, accountant=BudgetAccountant())
    dp_auc = dp_hist.sum() / len(y_true)

    return dp_auc
