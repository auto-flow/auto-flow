from autopipeline.pipeline.components.feature_engineer.encode.base import BaseEncoder

__all__ = ["BinaryEncoder"]


class BinaryEncoder(BaseEncoder):
    class__ = "BinaryEncoder"
    module__ = "category_encoders"
