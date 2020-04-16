from autoflow.pipeline.components.preprocessing.encode.base import BaseEncoder

__all__ = ["LabelEncoder"]


class LabelEncoder(BaseEncoder):
    class__ = "LabelEncoder"
    module__ = "autoflow.feature_engineer.encode.label_encode"
