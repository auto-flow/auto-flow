from autopipeline.pipeline.components.feature_engineer.select.from_model_base import SelectFromModelBase

__all__ = ["SelectFromModelClf"]


class SelectFromModelClf(SelectFromModelBase):
    classification_only = True
