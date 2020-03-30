from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["RepeatedEditedNearestNeighbours"]


class RepeatedEditedNearestNeighbours(AutoPLDataProcessAlgorithm):
    class__ = "RepeatedEditedNearestNeighbours"
    module__ = "imblearn.under_sampling"
