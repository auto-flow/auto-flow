from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["CondensedNearestNeighbour"]


class CondensedNearestNeighbour(AutoFlowDataProcessAlgorithm):
    class__ = "CondensedNearestNeighbour"
    module__ = "imblearn.under_sampling"
