from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["SMOTENC"]


class SMOTENC(AutoFlowDataProcessAlgorithm):
    class__ = "SMOTENC"
    module__ = "imblearn.over_sampling"
