from collections import Counter
from typing import List, Union

from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd


class GroupSeries(pd.Series):
    def __init__(self, *args, **kwargs):
        super(GroupSeries, self).__init__(*args, **kwargs)

    def __repr__(self):
        return Counter(self).__repr__().replace("Counter", "GroupSeries")

class FeatureGroup(TransformerMixin, BaseEstimator):
    def __init__(self, selected_group: str, feature_groups: Union[List, GroupSeries]):
        self.feature_groups = GroupSeries(feature_groups)
        self.selected_group = selected_group
        assert selected_group in feature_groups.to_list()

    def __validate_X(self,X):
        assert len(X.shape)==2
        assert X.shape[1]==self.feature_groups.shape[1]

    def fit(self, X, y=None):
        self.__validate_X(X)
        return self

    def transform(self, X):
        self.__validate_X(X)
        return X[:,self.feature_groups==self.selected_group]
