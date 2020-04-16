from autoflow.pipeline.components.preprocessing.encode.base import BaseEncoder

__all__ = ["CatBoostEncoder"]


class CatBoostEncoder(BaseEncoder):
    class__ = "CatBoostEncoder"
    module__ = "category_encoders"
    need_y = True
