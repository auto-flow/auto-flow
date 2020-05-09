from autoflow.pipeline.components.preprocessing.encode.base import BaseEncoder

__all__ = ["LabelEncoder"]


class LabelEncoder(BaseEncoder):
    class__ = "OrdinalEncoder"
    module__ = "category_encoders"
