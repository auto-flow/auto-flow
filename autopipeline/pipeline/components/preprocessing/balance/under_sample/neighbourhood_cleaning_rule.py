from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["NeighbourhoodCleaningRule"]


class NeighbourhoodCleaningRule(AutoPLDataProcessAlgorithm):
    class__ = "NeighbourhoodCleaningRule"
    module__ = "imblearn.under_sampling"
