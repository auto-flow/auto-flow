from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["AllKNN"]


class AllKNN(HyperFlowDataProcessAlgorithm):
    class__ = "AllKNN"
    module__ = "imblearn.under_sampling"
