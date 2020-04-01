from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["CondensedNearestNeighbour"]


class CondensedNearestNeighbour(HyperFlowDataProcessAlgorithm):
    class__ = "CondensedNearestNeighbour"
    module__ = "imblearn.under_sampling"
