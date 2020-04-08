from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm
from hyperflow.pipeline.components.utils import get_categorical_features_indices
from hyperflow.utils.data import arraylize

__all__ = ["LGBMClassifier"]


class LGBMClassifier(HyperFlowClassificationAlgorithm):
    class__ = "LGBMClassifier"
    module__ = "lightgbm"

    boost_model = True
    tree_model = True

    def core_fit(self, estimator, X, y=None, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None, columns_metadata=None):
        categorical_features_indices = get_categorical_features_indices(X, columns_metadata)
        X = arraylize(X)
        X_valid = arraylize(X_valid)
        if (X_valid is not None) and (y_valid is not None):
            eval_set = (X_valid, y_valid)
        else:
            eval_set = None
        return self.estimator.fit(
            X, y, categorical_feature=categorical_features_indices,
            eval_set=eval_set, verbose=False,
            early_stopping_rounds=self.hyperparams.get("early_stopping_rounds")
        )

    def before_pred_X(self, X):
        return arraylize(X)
