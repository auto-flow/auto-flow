from autopipeline.pipeline.components.feature_engineer.select.from_model_base import SelectFromModelBase

__all__ = ["SelectFromModelReg"]


class SelectFromModelReg(SelectFromModelBase):
    regression_only = True
