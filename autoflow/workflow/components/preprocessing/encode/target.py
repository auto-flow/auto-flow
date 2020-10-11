from autoflow.workflow.components.preprocessing.encode.base import BaseCategoryEncoders

__all__ = ["TargetEncoder"]


class TargetEncoder(BaseCategoryEncoders):
    class__ = "TargetEncoder"
    module__ = "category_encoders"
    need_y = True
