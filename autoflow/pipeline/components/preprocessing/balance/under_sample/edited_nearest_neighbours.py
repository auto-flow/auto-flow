from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["EditedNearestNeighbours"]


class EditedNearestNeighbours(AutoFlowDataProcessAlgorithm):
    class__ = "EditedNearestNeighbours"
    module__ = "imblearn.under_sampling"
