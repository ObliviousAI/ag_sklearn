import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from diffprivlib.tools.utils import nanmean

class DifferentiallyPrivateImputer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_values=np.nan, strategy='mean', fill_value=None, epsilon=1.0):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.epsilon = epsilon  # Privacy budget
        
    def fit(self, X, y=None):
        if self.strategy == 'most_frequent':
            self.statistics_ = [np.nan] * X.shape[1]  # Not used in most_frequent strategy
        elif self.strategy == 'constant':
            self.statistics_ = [self.fill_value] * X.shape[1]
        else:
            self.statistics_ = [nanmean(col) if np.issubdtype(col.dtype, np.number) else np.nan for col in X.T]
        return self
    
    def _impute_mean(self, col, missing_col):
        non_missing_values = col[~missing_col]
        col_mean = nanmean(non_missing_values)
        sensitivity = np.nanmax(np.abs(non_missing_values - col_mean))
        return col_mean, sensitivity
    
    def _impute_median(self, col, missing_col):
        non_missing_values = col[~missing_col]
        col_median = np.nanmedian(non_missing_values)
        sensitivity = np.nanmax(np.abs(non_missing_values - col_median))
        return col_median, sensitivity
    
    def transform(self, X):
        noisy_X = np.copy(X)
        missing_indices = np.isnan(X)
        
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            missing_col = missing_indices[:, col_idx]
            
            if np.any(missing_col):
                if self.strategy == 'mean':
                    col_mean, sensitivity = self._impute_mean(col, missing_col)
                elif self.strategy == 'median':
                    col_mean, sensitivity = self._impute_median(col, missing_col)
                elif self.strategy == 'most_frequent':
                    col_mean = self.statistics_[col_idx]  # Use most frequent value
                    sensitivity = 1  # Sensitivity for most_frequent is 1
                
                scale = sensitivity / self.epsilon
                laplace_noise = np.random.laplace(loc=0, scale=scale, size=np.sum(missing_col))
                
                # Impute missing values with noisy values
                noisy_X[missing_col, col_idx] = col_mean + laplace_noise
                
        return noisy_X
