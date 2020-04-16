from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["AllKNN"]


class AllKNN(AutoFlowDataProcessAlgorithm):
    class__ = "AllKNN"
    module__ = "imblearn.under_sampling"
