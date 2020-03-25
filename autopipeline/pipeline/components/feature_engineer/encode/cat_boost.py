from autopipeline.pipeline.components.feature_engineer.encode.base import BaseEncoder

__all__ = ["CatBoostEncoder"]


class CatBoostEncoder(BaseEncoder):
    class__ = "CatBoostEncoder"
    module__ = "category_encoders"
    need_y = True
