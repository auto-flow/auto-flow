from autoflow.workflow.components.preprocessing.select.base import SelectFromModelBase

__all__ = ["SelectFromModelReg"]


class SelectFromModelReg(SelectFromModelBase):
    regression_only = True
