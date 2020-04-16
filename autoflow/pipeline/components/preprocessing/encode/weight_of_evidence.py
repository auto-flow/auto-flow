from autoflow.pipeline.components.preprocessing.encode.base import BaseEncoder

__all__ = ["WOEEncoder"]


class WOEEncoder(BaseEncoder):
    class__ = "WOEEncoder"
    module__ = "category_encoders"
    need_y = True