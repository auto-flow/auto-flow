from collections import Counter
from copy import deepcopy

import pandas as pd


class GeneralDataFram(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        if "feat_grp" in kwargs:
            self.feat_grp = kwargs.pop("feat_grp")
        else:
            self.feat_grp = None
        super(GeneralDataFram, self).__init__(*args, **kwargs)
        if self.feat_grp is None:
            self.feat_grp = ["cat"] * self.shape[1]
        assert (len(self.feat_grp) == self.shape[1])
        self.feat_grp = pd.Series(self.feat_grp)
        self.origin_grp = deepcopy(self.feat_grp)

    def __repr__(self):
        return super(GeneralDataFram, self).__repr__() + "\n" + repr(Counter(self.feat_grp))

    def drop(
            self,
            labels=None,
            axis=0,
            index=None,
            columns=None,
            level=None,
            inplace=False,
            errors="raise",
    ):

        if axis == 1:
            columns_ = labels or columns
            assert columns_ is not None
            feat_grp = self.feat_grp[~self.columns.isin(columns_)]
            origin_grp = self.origin_grp[~self.columns.isin(columns_)]
        ret = super(GeneralDataFram, self).drop(
            labels=labels,
            axis=axis,
            index=index,
            columns=columns,
            level=level,
            inplace=inplace,
            errors=errors,
        )
        if axis == 1:
            if inplace:
                self.feat_grp = feat_grp
                self.origin_grp = origin_grp
            else:
                ret.feat_grp=feat_grp
                ret.origin_grp=origin_grp
        return ret
