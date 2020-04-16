import pandas as pd

from autoflow.pipeline.dataframe import GenericDataFrame


def is_nan_rejection(X: GenericDataFrame, y, nan_grp):
    selected_df = X.filter_feature_groups(nan_grp)
    deleted_index = selected_df.dropna().index
    if not isinstance(y, (pd.DataFrame, pd.Series)):
        y = pd.Series(y)
    y.index = selected_df.index
    return next(X.split([deleted_index], type="loc")), y[deleted_index].values


class DeleteNanRow():
    def __init__(self, nan_grp="lowR_nan"):
        self.nan_grp = nan_grp

    def fit_sample(self, X, y):
        return is_nan_rejection(X, y, self.nan_grp)
