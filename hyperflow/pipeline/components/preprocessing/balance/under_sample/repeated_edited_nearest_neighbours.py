from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["RepeatedEditedNearestNeighbours"]


class RepeatedEditedNearestNeighbours(HyperFlowDataProcessAlgorithm):
    class__ = "RepeatedEditedNearestNeighbours"
    module__ = "imblearn.under_sampling"
