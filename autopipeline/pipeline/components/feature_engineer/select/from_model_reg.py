from autopipeline.pipeline.components.feature_engineer.select.base import SelectFromModelBase

__all__ = ["SelectFromModelReg"]


class SelectFromModelReg(SelectFromModelBase):
    regression_only = True
