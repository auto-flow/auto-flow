import sklearn.feature_selection

from autoflow.pipeline.components.preprocessing.select.base import SelectPercentileBase

__all__ = ["SelectPercentileClassification"]


class SelectPercentileClassification(SelectPercentileBase):
    classification_only = True
    def get_default_name(self):
        return "chi2"

    def get_name2func(self):
        return {
            "chi2": sklearn.feature_selection.chi2,
            "f_classif": sklearn.feature_selection.f_classif,
            "mutual_info_classif": sklearn.feature_selection.mutual_info_classif
        }
