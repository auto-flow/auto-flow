from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["CondensedNearestNeighbour"]


class CondensedNearestNeighbour(AutoPLDataProcessAlgorithm):
    class__ = "CondensedNearestNeighbour"
    module__ = "imblearn.under_sampling"
