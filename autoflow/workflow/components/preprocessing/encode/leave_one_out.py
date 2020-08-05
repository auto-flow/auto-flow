from autoflow.workflow.components.preprocessing.encode.base import BaseCategoryEncoders

__all__ = ["LeaveOneOutEncoder"]


class LeaveOneOutEncoder(BaseCategoryEncoders):
    class__ = "LeaveOneOutEncoder"
    module__ = "category_encoders"
    need_y = True
