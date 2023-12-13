from sklearn import metrics
import numpy as np

def confusion_matrix(y_true, y_pred, epsilon=1.0,bounds=None, random_state=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    conf_matrix = metrics.confusion_matrix(y_true, y_pred)

    sensitivity = 1.0
    
    dp_conf_matrix = conf_matrix.copy()
    
    for i in range(dp_conf_matrix.shape[0]):
        for j in range(dp_conf_matrix.shape[1]):
            scale = sensitivity / epsilon
            laplace_noise = np.random.laplace(scale=scale, size=1)
            dp_conf_matrix[i, j] += laplace_noise
            
    return dp_conf_matrix
    
