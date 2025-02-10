import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class PassThroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns
        else:
            self._feature_names = None
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        if self._feature_names is not None:
            return self._feature_names
        return input_features

class ThresholdApplier(BaseEstimator, TransformerMixin):
    """
    Apply a threshold to the data.
    """
    def __init__(self, threshold=0.99, threshold_type='hard'):
        """
        Initializes the transformer with the threshold and threshold type.
        
        Parameters:
            threshold (float): The threshold value.
            threshold_type (str): 'hard' or 'soft'. 'hard' will set values below the threshold to 0.
                                             'soft' will subtract the threshold from all values.
        """
        self.threshold = threshold
        self.threshold_type = threshold_type
        self._feature_names = None
    
    def fit(self, X, y=None):
        """
        This transformer doesn't learn anything from the data, so the fit method just returns self.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns
        return self
    
    def transform(self, X):
        """
        Apply the threshold to the data.
        
        Parameters:
            X (pd.DataFrame): Input data.
            
        Returns:
            pd.DataFrame: Data with the threshold applied.
        """
        X_transformed = X.copy()

        # Check if threshold_type is valid
        if self.threshold_type not in ['hard', 'soft']:
            raise ValueError("threshold_type must be 'hard' or 'soft'")
        
        # Apply the threshold
        if self.threshold_type == 'hard':
            X_transformed = X_transformed.where(X_transformed >= self.threshold, 0)
        elif self.threshold_type == 'soft':
            X_transformed = X_transformed - self.threshold
            X_transformed = X_transformed.clip(lower=0)
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        if self._feature_names is not None:
            return self._feature_names
        return input_features

class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0, threshold=0.01):
        self.alpha = alpha
        self.threshold = threshold
        self.lasso = Lasso(alpha=self.alpha, max_iter=1000000)
        self.support_mask = None
        self.scaler = StandardScaler()
        self._feature_names = None

    def fit(self, X, y=None):
        # If X is DataFrame, store original col names
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns
            X_array = X.values  # for scaling / Lasso
        else:
            # Just store numeric range
            self._feature_names = np.arange(X.shape[1]).astype(str)
            X_array = X
        
        X_scaled = self.scaler.fit_transform(X_array)
        self.lasso.fit(X_scaled, y)
        self.support_mask = np.abs(self.lasso.coef_) > self.threshold
        # If all coefficients < threshold, keep them all (as you already do)
        if not any(self.support_mask):
            self.support_mask = np.array([True] * X.shape[1])
        
        return self

    def transform(self, X):
        # Because we stored _feature_names, let's maintain a DataFrame output
        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            X_array = X.values
        else:
            X_array = X
        
        X_scaled = self.scaler.transform(X_array)
        X_selected = X_scaled[:, self.support_mask]
        
        if is_df:
            # Filter the original column names to just the ones we kept
            selected_cols = np.array(self._feature_names)[self.support_mask]
            return pd.DataFrame(X_selected, columns=selected_cols, index=X.index)
        else:
            return X_selected

    def get_feature_names_out(self, input_features=None):
        # Return only the columns that survived
        return np.array(self._feature_names)[self.support_mask]

class RandomForestFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select features using Random Forest.
    """
    def __init__(self, n_estimators=100, threshold=0.001, n_jobs=1):
        """
        Initializes the transformer with the number of trees in the forest and the threshold
        for feature importances.
        
        Parameters:
            n_estimators (int): The number of trees in the forest.
            threshold (float): Feature importances below this threshold will be discarded.
        """
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=n_jobs)
        self._feature_names = None
    
    def fit(self, X, y=None):
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        self._is_df = isinstance(X, pd.DataFrame)
        if self._is_df:
            self._feature_names = X.columns
            X_array = X.values
        else:
            self._feature_names = np.arange(X.shape[1]).astype(str)
            X_array = X
        
        # Fit RandomForest
        self.model.fit(X_array, y)
        self.feature_importances_ = self.model.feature_importances_
        self.selected_features_ = self.feature_importances_ >= self.threshold
        return self
    
    def transform(self, X):
        # Check if X has the same number of features as the data used in fit
        if X.shape[1] != len(self.selected_features_):
            raise ValueError("X has a different number of features than the data used in fit")
        
        # Determine if input is DataFrame
        is_dataframe = isinstance(X, pd.DataFrame)
        
        if is_dataframe:
            X_array = X.values
        else:
            X_array = X

        X_selected = X_array[:, self.selected_features_]
        
        if is_dataframe:
            selected_columns = np.array(self._feature_names)[self.selected_features_]
            X_selected = pd.DataFrame(X_selected, columns=selected_columns, index=X.index)
        return X_selected
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self._feature_names)[self.selected_features_]
