from autoflow.pipeline.components.preprocessing.encode.base import BaseEncoder

__all__ = ["TargetEncoder"]


class TargetEncoder(BaseEncoder):
    class__ = "TargetEncoder"
    module__ = "category_encoders"
    need_y = True