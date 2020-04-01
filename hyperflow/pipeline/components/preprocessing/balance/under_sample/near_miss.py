from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["NearMiss"]


class NearMiss(HyperFlowDataProcessAlgorithm):
    class__ = "NearMiss"
    module__ = "imblearn.under_sampling"
