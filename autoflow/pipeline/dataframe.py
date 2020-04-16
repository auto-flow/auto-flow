from copy import deepcopy
from typing import List, Union

import numpy as np
import pandas as pd
from pandas._typing import FrameOrSeries
from pandas.core.generic import bool_t

from autoflow.utils.logging import get_logger

logger = get_logger(__name__)


class GenericDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        # self.
        if "feature_groups" in kwargs:
            feature_groups = kwargs.pop("feature_groups")
        else:
            feature_groups = None
        if "columns_metadata" in kwargs:
            columns_metadata = kwargs.pop("columns_metadata")
        else:
            columns_metadata = None
        super(GenericDataFrame, self).__init__(*args, **kwargs)
        if feature_groups is None:
            logger.debug("feature_groups is None, set it all to 'cat' feature group.")
            feature_groups = ["cat"] * self.shape[1]
        assert (len(feature_groups) == self.shape[1])
        self.set_feature_groups(pd.Series(feature_groups))
        if columns_metadata is None:
            columns_metadata = [{}] * self.shape[1]
        self.set_columns_metadata(pd.Series(columns_metadata))

    @property
    def feature_groups(self):
        return self.__dict__["feature_groups"]

    @property
    def columns_metadata(self):
        return self.__dict__["columns_metadata"]

    def set_feature_groups(self, feature_groups):
        self.__dict__["feature_groups"] = feature_groups

    def set_columns_metadata(self, columns_metadata):
        self.__dict__["columns_metadata"] = columns_metadata

    def __repr__(self):
        return super(GenericDataFrame, self).__repr__() + "\n" + "feature_groups: "+repr(list(self.feature_groups))

    def filter_feature_groups(self, feature_group: Union[List, str], copy=True, isin=True):  # , inplace=False
        if feature_group == "all":  # todo 用正则表达式判断
            feature_group = np.unique(self.feature_groups).tolist()
        # 用于过滤feature_groups
        if isinstance(feature_group, str):
            feature_group = [feature_group]
        if copy:
            result = deepcopy(self)
            result = GenericDataFrame(result, feature_groups=self.feature_groups,
                                      columns_metadata=self.columns_metadata)
        else:
            result = self
        loc = result.feature_groups.isin(feature_group)
        if not isin:
            loc = (~loc)
        result.set_feature_groups(result.feature_groups[loc])
        result.set_columns_metadata(result.columns_metadata[loc])
        loc_df = result.loc[:, result.columns[loc]]
        return GenericDataFrame(loc_df, feature_groups=result.feature_groups, columns_metadata=result.columns_metadata)

    def concat_two(self, df1, df2):
        assert isinstance(df1, GenericDataFrame)
        assert isinstance(df2, GenericDataFrame)

        new_df = pd.concat([df1, df2], axis=1)
        # todo: 杜绝重复列
        new_feature_groups = pd.concat([df1.feature_groups, df2.feature_groups], ignore_index=True)
        new_columns_metadata = pd.concat([df1.columns_metadata, df2.columns_metadata], ignore_index=True)
        return GenericDataFrame(new_df, feature_groups=new_feature_groups, columns_metadata=new_columns_metadata)

    def replace_feature_groups(self, old_feature_group: Union[List[str], str],
                               values: Union[np.ndarray, pd.DataFrame],
                               new_feature_group: Union[str, List[str], pd.Series],
                               new_columns_metadata: Union[str, List[dict], None, pd.Series] = None):
        if old_feature_group == "all":
            old_feature_group = np.unique(self.feature_groups).tolist()
        if isinstance(old_feature_group, str):
            old_feature_group = [old_feature_group]

        if new_columns_metadata is None:
            new_columns_metadata = [{}] * values.shape[1]
        assert len(new_columns_metadata) == values.shape[1]
        new_columns_metadata = pd.Series(new_columns_metadata)

        # 将 new_feature_groups 从str表达为list
        if isinstance(new_feature_group, str):
            new_feature_group = [new_feature_group] * values.shape[1]
        assert len(new_feature_group) == values.shape[1]
        new_feature_group = pd.Series(new_feature_group)

        # new_df 的 columns
        replaced_columns = self.columns[self.feature_groups.isin(old_feature_group)]
        if len(replaced_columns) == values.shape[1]:
            columns = replaced_columns
        else:
            columns = [f"{x}_{i}" for i, x in enumerate(new_feature_group)]

        # 开始构造df
        if isinstance(values, np.ndarray):
            values = pd.DataFrame(values, columns=columns)
        deleted_df = self.filter_feature_groups(old_feature_group, True, False)
        new_df = GenericDataFrame(values, feature_groups=new_feature_group,
                                  columns_metadata=new_columns_metadata)
        new_df.index = deleted_df.index
        return self.concat_two(deleted_df, new_df)

    def split(self, indexes, type="iloc"):
        assert type in ("loc", "iloc")
        for index in indexes:
            if type == "iloc":
                yield GenericDataFrame(self.iloc[index, :], feature_groups=self.feature_groups,
                                       columns_metadata=self.columns_metadata)
            elif type == "loc":
                yield GenericDataFrame(self.loc[index, :], feature_groups=self.feature_groups,
                                       columns_metadata=self.columns_metadata)

    def copy(self: FrameOrSeries, deep: bool_t = True) -> FrameOrSeries:
        return GenericDataFrame(super(GenericDataFrame, self).copy(deep=deep), feature_groups=self.feature_groups,
                                columns_metadata=self.columns_metadata)

    def __reduce__(self):
        result = super(GenericDataFrame, self).__reduce__()
        result[2].update({
            "feature_groups": self.feature_groups,
            "columns_metadata": self.columns_metadata
        })
        return result

    def __setstate__(self, state):
        self.set_feature_groups(state.pop("feature_groups"))
        self.set_columns_metadata(state.pop("columns_metadata"))
        super(GenericDataFrame, self).__setstate__(state)


if __name__ == '__main__':
    import logging

    df = pd.read_csv("/home/tqc/PycharmProjects/AutoFlow/examples/classification/train_classification.csv")
    suffix = ["num"] * 2 + ["cat"] * 2 + ["num"] * 5 + ["cat"] * 2
    feature_groups = ["id"] + suffix

    df2 = GenericDataFrame(df, feature_groups=feature_groups)
    # 测试1->2
    selected = df2.filter_feature_groups("id").values
    selected = np.hstack([selected, selected])
    df3 = df2.replace_feature_groups("id", selected, "id2")
    logging.info(df3)
    # 测试1->1
    selected = df2.filter_feature_groups("id").values
    selected = np.hstack([selected])
    df3 = df2.replace_feature_groups("id", selected, "id2")
    logging.info(df3)
    # 测试1->0
    selected = df2.filter_feature_groups("id").values
    selected = np.zeros([selected.shape[0], 0])
    df3 = df2.replace_feature_groups("id", selected, "id2")
    logging.info(df3)
