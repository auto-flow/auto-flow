from autoflow.workflow.components.preprocessing.select.base import SelectFromModelBase

__all__ = ["SelectFromModelClf"]


class SelectFromModelClf(SelectFromModelBase):
    classification_only = True
