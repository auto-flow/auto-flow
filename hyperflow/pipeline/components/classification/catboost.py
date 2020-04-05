from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm
from hyperflow.pipeline.components.utils import get_categorical_features_indices

__all__ = ["CatBoostClassifier"]


class CatBoostClassifier(HyperFlowClassificationAlgorithm):
    class__ = "CatBoostClassifier"
    module__ = "catboost"

    boost_model = True
    tree_model = True

    def core_fit(self, estimator, X, y=None, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feat_grp=None, origin_grp=None):
        categorical_features_indices = get_categorical_features_indices(X, origin_grp)
        if (X_valid is not None) and (y_valid is not None):
            eval_set = (X_valid, y_valid)
        else:
            eval_set = None
        return self.estimator.fit(
            X, y, cat_features=categorical_features_indices,
            eval_set=eval_set, silent=True
        )
