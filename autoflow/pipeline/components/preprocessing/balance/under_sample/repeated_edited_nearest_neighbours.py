from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["RepeatedEditedNearestNeighbours"]


class RepeatedEditedNearestNeighbours(AutoFlowDataProcessAlgorithm):
    class__ = "RepeatedEditedNearestNeighbours"
    module__ = "imblearn.under_sampling"
