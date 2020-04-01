from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["RandomOverSampler"]


class RandomOverSampler(HyperFlowDataProcessAlgorithm):
    class__ = "RandomOverSampler"
    module__ = "imblearn.over_sampling"
