import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder
from tabpfn import TabPFNClassifier


class TabpfnClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_ensemble_configurations: int = 32):
        self.n_ensemble_configurations = n_ensemble_configurations

    def fit(self, X, y):
        # Set tablevectorizer and tabpfn classifier
        # self.preprocessor_ = TableVectorizer(auto_cast=True, sparse_threshold=0)
        self.preprocessor_ = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )

        self.classifier_ = TabPFNClassifier(
            device="cpu", N_ensemble_configurations=self.n_ensemble_configurations
        )

        X = self.preprocessor_.fit_transform(X)

        if X.shape[1] > 100:
            n_components = np.min([X.shape[0], 100])
            self.pca_ = PCA(n_components=n_components, svd_solver="full")
            X = self.pca_.fit_transform(X)

        self.classifier_.fit(X=X, y=y)

        self.is_fitted_ = True

        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = self.preprocessor_.transform(X)
        if X.shape[1] > 100:
            X = self.pca_.transform(X)
        y_eval, _ = self.classifier_.predict(X=X, return_winning_probability=True)
        return y_eval

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        X = self.preprocessor_.transform(X)
        if X.shape[1] > 100:
            X = self.pca_.transform(X)
        _, p_eval = self.classifier_.predict(X=X, return_winning_probability=True)
        return p_eval
