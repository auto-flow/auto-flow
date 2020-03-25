from autopipeline.pipeline.components.feature_engineer.select.rfe_base import REF_Base

__all__ = ["RFE_Reg"]


class RFE_Reg(REF_Base):
    regression_only = True
