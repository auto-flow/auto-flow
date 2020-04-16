from autoflow.pipeline.components.preprocessing.encode.base import BaseEncoder

__all__ = ["BinaryEncoder"]


class BinaryEncoder(BaseEncoder):
    class__ = "BinaryEncoder"
    module__ = "category_encoders"
