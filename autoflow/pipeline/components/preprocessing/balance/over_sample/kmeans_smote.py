from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["KMeansSMOTE"]


class KMeansSMOTE(AutoFlowDataProcessAlgorithm):
    class__ = "KMeansSMOTE"
    module__ = "imblearn.over_sampling"
