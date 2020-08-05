from autoflow.workflow.components.preprocessing.encode.base import BaseCategoryEncoders

__all__ = ["OneHotEncoder"]


class OneHotEncoder(BaseCategoryEncoders):
    class__ = "OneHotEncoder"
    module__ = "category_encoders"
