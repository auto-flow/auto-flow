from autopipeline.pipeline.components.feature_engineer.select.base import SelectFromModelBase

__all__ = ["SelectFromModelClf"]


class SelectFromModelClf(SelectFromModelBase):
    classification_only = True
