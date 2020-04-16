from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["NeighbourhoodCleaningRule"]


class NeighbourhoodCleaningRule(AutoFlowDataProcessAlgorithm):
    class__ = "NeighbourhoodCleaningRule"
    module__ = "imblearn.under_sampling"
