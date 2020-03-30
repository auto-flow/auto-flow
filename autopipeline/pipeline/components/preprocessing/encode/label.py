from autopipeline.pipeline.components.preprocessing.encode.base import BaseEncoder

__all__ = ["LabelEncoder"]


class LabelEncoder(BaseEncoder):
    class__ = "LabelEncoder"
    module__ = "autopipeline.feature_engineer.encode.label_encode"
