
from diffprivlib.Classification_Metrics.confusion_matrix import confusion_matrix

def average_precision(y_true, y_pred, epsilon=1.0, bounds=None, random_state=None):

    conf_matrix=confusion_matrix(y_true, y_pred, epsilon=epsilon,bounds=bounds, random_state=random_state)

    tp = conf_matrix[1, 1]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    avg_precision = precision * recall if (precision + recall) > 0 else 0

    return avg_precision
