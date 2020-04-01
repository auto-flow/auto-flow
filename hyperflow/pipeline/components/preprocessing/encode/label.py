from hyperflow.pipeline.components.preprocessing.encode.base import BaseEncoder

__all__ = ["LabelEncoder"]


class LabelEncoder(BaseEncoder):
    class__ = "LabelEncoder"
    module__ = "hyperflow.feature_engineer.encode.label_encode"
