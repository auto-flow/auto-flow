from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm
from autoflow.workflow.components.utils import get_categorical_features_indices
from autoflow.utils.data import to_array

__all__ = ["LGBMClassifier"]


class LGBMClassifier(AutoFlowClassificationAlgorithm):
    class__ = "LGBMClassifier"
    module__ = "lightgbm"

    boost_model = True
    tree_model = True

    def core_fit(self, estimator, X, y=None, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None):
        categorical_features_indices = get_categorical_features_indices(X)
        if (X_valid is not None) and (y_valid is not None):
            eval_set = (X_valid, y_valid)
        else:
            eval_set = None
        early_stopping_rounds=self.hyperparams.get("early_stopping_rounds")
        if eval_set is None:
            early_stopping_rounds=None
        return self.component.fit(
            X, y, categorical_feature=categorical_features_indices,
            eval_set=eval_set, verbose=False,
            early_stopping_rounds=early_stopping_rounds
        )


