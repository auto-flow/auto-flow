from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["RandomUnderSampler"]


class RandomUnderSampler(AutoFlowDataProcessAlgorithm):
    class__ = "RandomUnderSampler"
    module__ = "imblearn.under_sampling"
