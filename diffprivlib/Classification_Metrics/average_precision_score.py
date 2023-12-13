from sklearn import metrics
import numpy as np
from diffprivlib.tools.utils import mean
from diffprivlib.accountant import BudgetAccountant

def average_precision_score(y_true, y_score, epsilon=1.0, bounds=None, random_state=None):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    avg_precision = metrics.average_precision_score(y_true, y_score)

    accountant = BudgetAccountant()
    dp_mean = mean(y_score, bounds=bounds, epsilon=epsilon, random_state=random_state, accountant=accountant)

    return avg_precision + dp_mean
