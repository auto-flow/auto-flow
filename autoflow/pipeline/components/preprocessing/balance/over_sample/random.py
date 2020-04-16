from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["RandomOverSampler"]


class RandomOverSampler(AutoFlowDataProcessAlgorithm):
    class__ = "RandomOverSampler"
    module__ = "imblearn.over_sampling"
