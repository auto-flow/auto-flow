from autopipeline.pipeline.components.feature_engineer.select.rfe_base import REF_Base

__all__ = ["RFE_Clf"]


class RFE_Clf(REF_Base):
    classification_only = True
