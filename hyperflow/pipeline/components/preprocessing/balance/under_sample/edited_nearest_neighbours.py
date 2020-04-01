from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["EditedNearestNeighbours"]


class EditedNearestNeighbours(HyperFlowDataProcessAlgorithm):
    class__ = "EditedNearestNeighbours"
    module__ = "imblearn.under_sampling"
