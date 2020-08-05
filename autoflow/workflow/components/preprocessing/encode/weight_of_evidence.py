from autoflow.workflow.components.preprocessing.encode.base import BaseCategoryEncoders

__all__ = ["WOEEncoder"]


class WOEEncoder(BaseCategoryEncoders):
    class__ = "WOEEncoder"
    module__ = "category_encoders"
    need_y = True
