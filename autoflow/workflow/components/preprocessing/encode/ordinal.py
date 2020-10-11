import numpy as np

from autoflow.workflow.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__ = ["OrdinalEncoder"]


class OrdinalEncoder(AutoFlowFeatureEngineerAlgorithm):
    class__ = "OrdinalEncoder"
    module__ = "sklearn.preprocessing"
    # cache_intermediate = True

    def get_category(self, Xs, column):
        Xs = [arg for arg in Xs if arg is not None]
        uniques = []
        for X in Xs:
            uniques.extend(X[column].unique().tolist())
        return sorted(list(set(uniques)))

    def after_process_estimator(self, estimator, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None,
                                y_test=None):
        categories = []
        for column in X_train.columns:
            if X_train[column].dtype.name == "category":
                category = X_train[column].cat.categories.to_list()
            else:
                category = self.get_category([X_train, X_valid, X_test], column=column)
            categories.append(category)
        estimator.categories = categories
        estimator.dtype = np.int
        return estimator
