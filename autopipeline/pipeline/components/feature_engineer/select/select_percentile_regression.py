from autopipeline.pipeline.components.feature_engineer.select.select_percentile import SelectPercentileBase
import sklearn.feature_selection

class SelectPercentileRegression(SelectPercentileBase):
    def get_default_name(self):
        return "f_regression"

    def get_name2func(self):
        return {
            "f_regression": sklearn.feature_selection.f_regression,
            "mutual_info": sklearn.feature_selection.mutual_info_regression,
        }
