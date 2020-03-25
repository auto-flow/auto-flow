from copy import deepcopy
from typing import List, Union

import numpy as np
import pandas as pd
from pandas._typing import FrameOrSeries
from pandas.core.generic import bool_t


class GenericDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        if "feat_grp" in kwargs:
            feat_grp = kwargs.pop("feat_grp")
        else:
            feat_grp = None
        if "origin_grp" in kwargs:
            origin_grp = kwargs.pop("origin_grp")
        else:
            origin_grp = None
        super(GenericDataFrame, self).__init__(*args, **kwargs)
        if feat_grp is None:
            feat_grp = ["cat"] * self.shape[1]
        assert (len(feat_grp) == self.shape[1])
        self.set_feat_grp(pd.Series(feat_grp))
        if origin_grp is None:
            origin_grp = deepcopy(feat_grp)
        self.set_origin_grp(pd.Series(origin_grp))

    @property
    def feat_grp(self):
        # return self.feat_grp_
        return self.__dict__["feat_grp"]

    @property
    def origin_grp(self):
        # return self.origin_grp_
        return self.__dict__["origin_grp"]

    def set_feat_grp(self, feat_grp):
        dict_={"feat_grp":feat_grp}
        if not self._metadata:
            self._metadata.append(dict_)
        else:
            se

    def set_origin_grp(self, origin_grp):
        # self.origin_grp_ = origin_grp
        self.__dict__["origin_grp"] = origin_grp

    def __repr__(self):
        return super(GenericDataFrame, self).__repr__() + "\n" + repr((self.feat_grp))

    def filter_feat_grp(self, feat_grp: Union[List, str], copy=True, isin=True):  # , inplace=False
        # 用于过滤feat_grp
        if isinstance(feat_grp, str):
            feat_grp = [feat_grp]
        if copy:
            ret = deepcopy(self)
            ret = GenericDataFrame(ret, feat_grp=self.feat_grp, origin_grp=self.origin_grp)
        else:
            ret = self
        loc = ret.feat_grp.isin(feat_grp)
        if not isin:
            loc = (~loc)
        ret.set_feat_grp(ret.feat_grp[loc])
        ret.set_origin_grp(ret.origin_grp[loc])
        loc_df = ret.loc[:, ret.columns[loc]]
        return GenericDataFrame(loc_df, feat_grp=ret.feat_grp, origin_grp=ret.origin_grp)

    def concat_two(self, df1, df2):
        assert isinstance(df1, GenericDataFrame)
        assert isinstance(df2, GenericDataFrame)

        new_df = pd.concat([df1, df2], axis=1)
        # todo: 杜绝重复列
        new_feat_grp = pd.concat([df1.feat_grp, df2.feat_grp], ignore_index=True)
        new_origin_grp = pd.concat([df1.origin_grp, df2.origin_grp], ignore_index=True)
        return GenericDataFrame(new_df, feat_grp=new_feat_grp, origin_grp=new_origin_grp)

    def replace_feat_grp(self, old_feat_grp: Union[List, str],
                         values: np.ndarray,
                         new_feat_grp: Union[str, List, pd.Series],
                         new_origin_grp: Union[str, List, None, pd.Series] = None):
        if isinstance(old_feat_grp, str):
            old_feat_grp = [old_feat_grp]
        # 如果参数new_origin_grp为None，根据长度是否改变对new_origin_grp进行赋值
        if new_origin_grp is None:
            selected_origin_grp = self.origin_grp[self.feat_grp.isin(old_feat_grp)]
            if len(selected_origin_grp) == values.shape[1]:
                new_origin_grp = deepcopy(selected_origin_grp)
            else:
                unique = pd.unique(selected_origin_grp)
                new_origin_grp = str(unique[0])
        # 将new_origin_grp从str表达为list
        if isinstance(new_origin_grp, str):
            # assert isinstance(new_origin_grp, str)
            new_origin_grp = [new_origin_grp] * values.shape[1]
        else:
            assert len(new_origin_grp) == values.shape[1]

        # 将 new_feat_grp 从str表达为list
        if isinstance(new_feat_grp, str):
            new_feat_grp = [new_feat_grp] * values.shape[1]
        # new_df 的 columns
        replaced_columns = self.columns[self.feat_grp.isin(old_feat_grp)]
        if len(replaced_columns) == values.shape[1]:
            columns = replaced_columns
        else:
            columns = [f"{x}_{i}" for i, x in enumerate(new_feat_grp)]
        # 开始构造df
        if isinstance(values, np.ndarray):
            values = pd.DataFrame(values, columns=columns)
        deleted_df = self.filter_feat_grp(old_feat_grp, True, False)
        new_df = GenericDataFrame(values, feat_grp=new_feat_grp,
                                  origin_grp=new_origin_grp)
        new_df.index = deleted_df.index
        return self.concat_two(deleted_df, new_df)

    def split(self, indexes):
        for index in indexes:
            yield GenericDataFrame(self.iloc[index, :].reset_index(drop=True), feat_grp=self.feat_grp,
                                   origin_grp=self.origin_grp)

    def copy(self: FrameOrSeries, deep: bool_t = True) -> FrameOrSeries:
        return GenericDataFrame(super(GenericDataFrame, self).copy(deep=deep), feat_grp=self.feat_grp,
                                origin_grp=self.origin_grp)

    def __reduce__(self):
        result = super(GenericDataFrame, self).__reduce__()
        result[2].update({
            "feat_grp": self.feat_grp,
            "origin_grp": self.origin_grp
        })
        return result

    def __setstate__(self, state):
        self.set_feat_grp(state.pop("feat_grp"))
        self.set_origin_grp(state.pop("origin_grp"))
        super(GenericDataFrame, self).__setstate__(state)


if __name__ == '__main__':
    import pickle
    df = pd.read_csv("/home/tqc/PycharmProjects/auto-pipeline/examples/classification/train_classification.csv")
    suffix = ["num"] * 2 + ["cat"] * 2 + ["num"] * 5 + ["cat"] * 2
    feat_grp = ["id"] + suffix

    df2 = GenericDataFrame(df, feat_grp=feat_grp)
    df=deepcopy(df2)
    print(df)
    print(pickle.loads(pickle.dumps(df)))
    # 测试1->2
    selected = df2.filter_feat_grp("id").values
    selected = np.hstack([selected, selected])
    df3 = df2.replace_feat_grp("id", selected, "id2")
    print(df3)
    # 测试1->1
    selected = df2.filter_feat_grp("id").values
    selected = np.hstack([selected])
    df3 = df2.replace_feat_grp("id", selected, "id2")
    print(df3)
    selected = df2.filter_feat_grp("id").values
    selected = np.zeros([selected.shape[0], 0])
    df3 = df2.replace_feat_grp("id", selected, "id2")
    print(df3)
