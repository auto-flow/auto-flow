from autoflow.workflow.components.preprocessing.encode.base import BaseCategoryEncoders

__all__ = ["BinaryEncoder"]


class BinaryEncoder(BaseCategoryEncoders):
    class__ = "BinaryEncoder"
    module__ = "category_encoders"
