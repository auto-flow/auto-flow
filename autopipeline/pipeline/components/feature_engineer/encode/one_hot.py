from autopipeline.pipeline.components.feature_engineer.encode.base import BaseEncoder

__all__ = ["OneHotEncoder"]


class OneHotEncoder(BaseEncoder):
    class__ = "OneHotEncoder"
    module__ = "category_encoders"
