from autoflow.pipeline.components.preprocessing.select.base import SelectFromModelBase

__all__ = ["SelectFromModelClf"]


class SelectFromModelClf(SelectFromModelBase):
    classification_only = True
