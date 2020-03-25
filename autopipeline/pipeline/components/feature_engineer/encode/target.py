from autopipeline.pipeline.components.feature_engineer.encode.base import BaseEncoder

__all__ = ["TargetEncoder"]


class TargetEncoder(BaseEncoder):
    class__ = "TargetEncoder"
    module__ = "category_encoders"
    need_y = True