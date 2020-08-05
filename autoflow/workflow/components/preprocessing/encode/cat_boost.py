from autoflow.workflow.components.preprocessing.encode.base import BaseCategoryEncoders

__all__ = ["CatBoostEncoder"]


class CatBoostEncoder(BaseCategoryEncoders):
    class__ = "CatBoostEncoder"
    module__ = "category_encoders"
    need_y = True
