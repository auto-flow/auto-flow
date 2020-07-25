from autoflow.workflow.components.preprocessing.encode.base import BaseEncoder

__all__ = ["OneHotEncoder"]


class OneHotEncoder(BaseEncoder):
    class__ = "OneHotEncoder"
    module__ = "autoflow.feature_engineer.encode"
