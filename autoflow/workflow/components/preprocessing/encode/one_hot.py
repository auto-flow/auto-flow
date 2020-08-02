from autoflow.workflow.components.preprocessing.encode.base import BaseEncoder

__all__ = ["OneHotEncoder"]


class OneHotEncoder(BaseEncoder):
    class__ = "CombineRare"
    module__ = "category_encoders"
