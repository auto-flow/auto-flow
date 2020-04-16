import sklearn.feature_selection

from autoflow.pipeline.components.preprocessing.select.base import SelectPercentileBase

__all__ = ["SelectPercentileRegression"]


class SelectPercentileRegression(SelectPercentileBase):
    regression_only = True
    def get_default_name(self):
        return "f_regression"

    def get_name2func(self):
        return {
            "f_regression": sklearn.feature_selection.f_regression,
            "mutual_info_regression": sklearn.feature_selection.mutual_info_regression,
        }
