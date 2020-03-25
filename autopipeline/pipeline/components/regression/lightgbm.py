from autopipeline.pipeline.components.classification_base import AutoPLClassificationAlgorithm
from autopipeline.pipeline.components.utils import get_categorical_features_indices
from autopipeline.utils.data import arraylize

__all__ = ["LGBMRegressor"]


class LGBMRegressor(AutoPLClassificationAlgorithm):
    class__ = "LGBMRegressor"
    module__ = "lightgbm"

    boost_model = True
    tree_model = True

    def _fit(self, estimator, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None,
             y_test=None, feat_grp=None, origin_grp=None):
        categorical_features_indices = get_categorical_features_indices(X_train,origin_grp)
        X_train = arraylize(X_train)
        X_valid = arraylize(X_valid)
        if (X_valid is not None) and (y_valid is not None):
            eval_set = (X_valid, y_valid)
        else:
            eval_set = None
        self.estimator.fit(
            X_train, y_train, categorical_feature=categorical_features_indices,
            eval_set=eval_set, verbose=False
        )

    def before_pred_X(self,X):
        return arraylize(X)