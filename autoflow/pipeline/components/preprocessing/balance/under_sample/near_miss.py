from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["NearMiss"]


class NearMiss(AutoFlowDataProcessAlgorithm):
    class__ = "NearMiss"
    module__ = "imblearn.under_sampling"
