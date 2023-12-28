from sklearn import metrics
import numpy as np
from opendp.mod import enable_features
from opendp.measurements import  make_base_laplace
from opendp.domains import atom_domain
from opendp.metrics import absolute_distance

def confusion_matrix(y_true, y_pred, epsilon=1.0,bounds=None, random_state=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    conf_matrix = metrics.confusion_matrix(y_true, y_pred)

    sensitivity = 1.0
    
    dp_conf_matrix = conf_matrix.copy()
    
    for i in range(dp_conf_matrix.shape[0]):
        for j in range(dp_conf_matrix.shape[1]):
            # Enable advanced features for OpenDP
            enable_features("contrib")

            laplace_scale = sensitivity / epsilon

            input_space = atom_domain(T=float), absolute_distance(T=float)
            base_lap = make_base_laplace(*input_space, scale=laplace_scale)    

            dp_conf_matrix[i][j] = base_lap( dp_conf_matrix[i, j])
            
    return dp_conf_matrix
    
