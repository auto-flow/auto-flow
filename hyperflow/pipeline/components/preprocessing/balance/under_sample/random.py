from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["RandomUnderSampler"]


class RandomUnderSampler(HyperFlowDataProcessAlgorithm):
    class__ = "RandomUnderSampler"
    module__ = "imblearn.under_sampling"
