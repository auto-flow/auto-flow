from autoflow.workflow.components.preprocessing.select.base import REF_Base

__all__ = ["RFE_Reg"]


class RFE_Reg(REF_Base):
    regression_only = True
