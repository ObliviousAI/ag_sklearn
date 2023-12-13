from sklearn import metrics
import numpy as np
from diffprivlib.tools.utils import mean
from diffprivlib.accountant import BudgetAccountant

def f1_score(y_true, y_pred, epsilon=1.0, bounds=None, random_state=None):
    # Convert labels to numpy arrays if they're not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate F1 score using the regular function
    f1 = metrics.f1_score(y_true, y_pred)

    # Calculate the differentially private mean of y_pred
    dp_mean = mean(y_pred, bounds=bounds, epsilon=epsilon, random_state=random_state, accountant=BudgetAccountant())

    # Introduce Laplace noise for differential privacy
    sensitivity = 1.0  # Placeholder sensitivity, adjust based on your data
    scale = sensitivity / epsilon
    laplace_noise = np.random.laplace(scale=scale, size=1)

    # Adjust the F1 score using dp_mean and Laplace noise
    dp_f1 = f1 + dp_mean + laplace_noise  # You can adjust the combination based on your requirements

    return dp_f1
