from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["KMeansSMOTE"]


class KMeansSMOTE(HyperFlowDataProcessAlgorithm):
    class__ = "KMeansSMOTE"
    module__ = "imblearn.over_sampling"
