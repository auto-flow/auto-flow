from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["EditedNearestNeighbours"]


class EditedNearestNeighbours(AutoPLDataProcessAlgorithm):
    class__ = "EditedNearestNeighbours"
    module__ = "imblearn.under_sampling"
