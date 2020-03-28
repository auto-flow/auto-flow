from autopipeline.pipeline.components.feature_engineer.encode.base import BaseEncoder

__all__ = ["LabelEncoder"]


class LabelEncoder(BaseEncoder):
    class__ = "LabelEncoder"
    module__ = "autopipeline.feature_engineer.encode.label_encode"
