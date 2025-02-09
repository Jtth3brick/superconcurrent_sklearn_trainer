import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# TODO: enable feature pass through after transform for feature importance

class PassThroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X

    def get_params(self):
        return None

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
        self.non_threshold_cols = ['Age', 'Gender', 'sum_known_pathogenic_bacteria', 'sum_known_pathogenic_eukaryota', 'sum_known_pathogenic_viruses']
    
    def fit(self, X, y=None):
        """
        This transformer doesn't learn anything from the data, so the fit method just returns self.
        """
        return self
    
    def transform(self, X):
        """
        Apply the threshold to the data.
        
        Parameters:
            X (pd.DataFrame): Input data.
            
        Returns:
            pd.DataFrame: Data with the threshold applied.
        """
        # Check if threshold_type is valid
        if self.threshold_type not in ['hard', 'soft']:
            raise ValueError("threshold_type must be 'hard' or 'soft'")
        
        # Create a copy of the DataFrame
        X_transformed = X.copy()
        
        # Columns to apply the threshold
        threshold_cols = [col for col in X.columns if col not in self.non_threshold_cols]
        
        # Apply the threshold
        if self.threshold_type == 'hard':
            X_transformed[threshold_cols] = X_transformed[threshold_cols].where(X_transformed[threshold_cols] >= self.threshold, 0)
        elif self.threshold_type == 'soft':
            X_transformed[threshold_cols] = X_transformed[threshold_cols] - self.threshold
            X_transformed[threshold_cols] = X_transformed[threshold_cols].clip(lower=0)
        
        return X_transformed

    def set_params(self, **params):
        """
        Set the parameters of the ThresholdApplier.

        Parameters:
            params (dict): A dictionary of parameters to set.

        Returns:
            self
        """
        for param, value in params.items():
            setattr(self, param, value)
        
        return self
    
    def get_params(self, deep=True):
        """
        Get the parameters of the ThresholdApplier.
        
        Parameters:
            deep (bool): If True, return a deep copy of the parameters.
            
        Returns:
            dict: A dictionary of the ThresholdApplier parameters.
        """
        return {'threshold': self.threshold, 'threshold_type': self.threshold_type}

class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0, threshold=0.01):
        self.alpha = alpha
        self.threshold = threshold
        self.lasso = Lasso(alpha=self.alpha, max_iter=1000000)
        self.support_mask = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        X = self.scaler.fit_transform(X)
        self.lasso.fit(X, y)
        self.support_mask = np.abs(self.lasso.coef_) > self.threshold
        if not any(self.support_mask):
            self.support_mask = np.array([True] * X.shape[1])
        
        return self

    def transform(self, X):
        if hasattr(X, 'loc'):
            return X.loc[:, self.support_mask]
        else:
            return X[:, self.support_mask]

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
    
    def fit(self, X, y=None):
        """
        Fits the Random Forest model and selects the features.
        
        Parameters:
            X (np.ndarray or pd.DataFrame): Input data.
            y (np.ndarray, pd.Series, or list): Target values.
            
        Returns:
            self
        """
        # Check if X and y are not None
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        # Fit the Random Forest model
        self.model.fit(X, y)
        
        # Get the feature importances from the model
        self.feature_importances_ = self.model.feature_importances_
        
        # Create a mask for features above the threshold
        self.selected_features_ = self.feature_importances_ >= self.threshold
        
        return self
    
    def transform(self, X):
        """
        Removes the features not selected by the Random Forest model.

        Parameters:
            X (np.ndarray or pd.DataFrame): Input data.

        Returns:
            np.ndarray or pd.DataFrame: Data with only the features selected by the Random Forest model.
        """
        # Check if X has the same number of features as the data used in fit
        if X.shape[1] != len(self.selected_features_):
            raise ValueError("X has a differXent number of features than the data used in fit")
        
        # Determine if input is DataFrame
        is_dataframe = isinstance(X, pd.DataFrame)
        
        # Convert input data to numpy array if it's DataFrame
        if is_dataframe:
            columns = X.columns
            X = X.values
        
        # Remove the features not selected by the Random Forest model
        if sum(self.selected_features_) > 0:
            X = X[:, self.selected_features_]

            # If original input was DataFrame, convert back to DataFrame
            if is_dataframe:
                # Create a mask for the column names
                selected_columns = np.array(columns)[self.selected_features_]
                X = pd.DataFrame(X, columns=selected_columns)
        
        return X
    
    def get_params(self, deep=True):
        """
        Get the parameters of the RandomForestFeatureSelector.
        
        Parameters:
            deep (bool): If True, return a deep copy of the parameters.
            
        Returns:
            dict: A dictionary of the RandomForestFeatureSelector parameters.
        """
        return {'n_estimators': self.n_estimators, 'threshold': self.threshold}
