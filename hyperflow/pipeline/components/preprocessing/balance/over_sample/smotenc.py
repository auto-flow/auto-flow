from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["SMOTENC"]


class SMOTENC(HyperFlowDataProcessAlgorithm):
    class__ = "SMOTENC"
    module__ = "imblearn.over_sampling"
