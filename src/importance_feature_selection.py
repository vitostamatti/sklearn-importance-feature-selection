import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import KFold
from sklearn.base import clone

class ImportanceFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(
            self, 
            estimator, 
            norm:bool=True, 
            rand:bool=True, 
            exp:bool=True, 
            choice:bool=True, 
            binom:bool=True, 
            cv:bool=None):
        
        self.estimator = estimator
        self.norm = norm
        self.rand = rand
        self.exp = exp
        self.choice = choice
        self.binom = binom
        self.cv = cv

        if not any([norm,rand,exp,choice,binom]):
            raise Exception("At least one of random features must be True")

    
    def generate_random_features(self, X):

        size=X.shape[0]
        index=X.index
        random_features = []

        if self.norm:
            x_norm = pd.DataFrame(np.random.normal(size=(size,1)), columns=['NOISE_x_norm'], index=index) 
            random_features.append(x_norm)
        if self.rand:
            x_rand = pd.DataFrame(np.random.random(size=(size,1)), columns=['NOISE_x_rand'], index=index)
            random_features.append(x_rand)
        if self.exp:
            x_exp = pd.DataFrame(np.random.exponential(size=(size,1)), columns=['NOISE_x_exp'], index=index)
            random_features.append(x_exp)
        if self.choice:
            x_choice = pd.DataFrame(np.random.choice([i for i in range(10)], size=(size,1)), columns=['NOISE_x_choice'], index=index)
            random_features.append(x_choice)
        if self.binom:
            x_binom = pd.DataFrame(np.random.binomial(n=1, p=0.7, size=(size,1)), columns=['NOISE_x_binom'], index=index)
            random_features.append(x_binom)

        X_random = pd.concat(random_features, axis=1)

        return X_random


    def compute_cv_feature_importances(self, estimator, X, y, n_splits=3):
        
        kf = KFold(n_splits=n_splits)

        feature_importances = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            estimator_ = clone(estimator)

            estimator_.fit(X_train, y_train)

            feature_importances.append(
                estimator_.feature_importances_
            )
        
        return np.mean(feature_importances, axis=0)


    def fit(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise Exception("X must be a pandas DataFrame object")

        X_random = self.generate_random_features(X)

        X_noise = pd.concat([X, X_random], axis=1)
        
        if self.cv:
            feature_importances = self.compute_cv_feature_importances(self.estimator, X_noise, y)
            self.estimator.fit(X_noise, y)
        else:
            self.estimator.fit(X_noise, y)
            feature_importances = self.estimator.feature_importances_

        feature_names = X_noise.columns.to_list()

        self.feature_importances_ = pd.DataFrame(
            feature_importances,
            index=feature_names,
            columns=['feature_importance']
        ).sort_values('feature_importance', ascending=False)

        random_feature_names = [f for f in feature_names if 'NOISE' in f]
        random_feature_idx = [feature_names.index(f) for f in feature_names if 'NOISE' in f]

        threshold = self.feature_importances_.loc[random_feature_names,:].values.max()

        selected_features_importance_ = self.feature_importances_[
            self.feature_importances_['feature_importance']>threshold
            ]

        self.selected_features_names = selected_features_importance_.index.to_list()
        
        self.selected_features_idx = [feature_names.index(f) for f in feature_names if f in self.selected_features_names]

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:,self.selected_features_idx]
        else:
            return X[self.selected_features_idx]
         