from sklearn import metrics
import numpy as np
from diffprivlib.tools.utils import mean
from diffprivlib.accountant import BudgetAccountant
# Define differentially private mean functions here

def accuracy_score(y_true, y_pred, epsilon=1.0, bounds=None, random_state=None):
    # Convert labels to numpy arrays if they're not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate accuracy using the regular accuracy_score function
    accuracy = metrics.accuracy_score(y_true, y_pred)

    # Add differential privacy to the accuracy calculation
    dp_accuracy = mean(np.equal(y_true, y_pred), bounds=bounds, epsilon=epsilon, random_state=random_state, accountant=BudgetAccountant())

    return dp_accuracy
