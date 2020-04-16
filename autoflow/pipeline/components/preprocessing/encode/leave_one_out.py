from autoflow.pipeline.components.preprocessing.encode.base import BaseEncoder

__all__ = ["LeaveOneOutEncoder"]


class LeaveOneOutEncoder(BaseEncoder):
    class__ = "LeaveOneOutEncoder"
    module__ = "category_encoders"
    need_y = True