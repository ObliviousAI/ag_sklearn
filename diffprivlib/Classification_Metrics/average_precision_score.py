from sklearn.metrics import average_precision_score
from opendp.mod import enable_features
from opendp.measurements import  make_base_laplace
from opendp.domains import atom_domain
from opendp.metrics import absolute_distance

def average_precision(y_true, y_pred, epsilon=1.0):

    output = average_precision_score(y_true, y_pred)

    # Enable advanced features for OpenDP
    enable_features("contrib")

    sensitivity=1

    laplace_scale = sensitivity / epsilon

    input_space = atom_domain(T=float), absolute_distance(T=float)
    base_lap = make_base_laplace(*input_space, scale=laplace_scale)    

    output = base_lap(output)

    return output

    



    
