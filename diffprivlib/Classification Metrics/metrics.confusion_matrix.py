from sklearn import metrics
import numpy as np
from diffprivlib.tools.utils import mean
from diffprivlib.accountant import BudgetAccountant

def confusion_matrix(y_true, y_pred, epsilon=1.0,bounds=None, random_state=None):
    # Convert labels to numpy arrays if they're not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate confusion matrix using the regular function
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)

    sensitivity = 1.0  # Placeholder sensitivity, adjust based on your data
    
    # Get the differentially private mean of y_pred
    dp_mean = mean(y_pred, bounds=bounds, epsilon=epsilon, random_state=random_state, accountant=BudgetAccountant())
    
    # Add differential privacy to the confusion matrix calculation
    dp_conf_matrix = conf_matrix.copy()  # Start with a copy of the regular confusion matrix
    
    # Introduce differential privacy logic to perturb the confusion matrix
    for i in range(dp_conf_matrix.shape[0]):
        for j in range(dp_conf_matrix.shape[1]):
            # Perturb each element of the confusion matrix using Laplace mechanism for differential privacy
            scale = sensitivity / epsilon
            laplace_noise = np.random.laplace(scale=scale, size=1)
            dp_conf_matrix[i, j] += laplace_noise + dp_mean  # Incorporate dp_mean here
            
    return dp_conf_matrix
    
