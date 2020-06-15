from autoflow.workflow.components.preprocessing.encode.base import BaseEncoder

__all__ = ["OrdinalEncoder"]


class OrdinalEncoder(BaseEncoder):
    class__ = "OrdinalEncoder"
    module__ = "category_encoders"
