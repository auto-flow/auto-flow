from autoflow.pipeline.components.preprocessing.select.base import REF_Base

__all__ = ["RFE_Clf"]


class RFE_Clf(REF_Base):
    classification_only = True
