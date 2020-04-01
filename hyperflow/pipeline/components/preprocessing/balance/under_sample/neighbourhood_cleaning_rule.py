from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["NeighbourhoodCleaningRule"]


class NeighbourhoodCleaningRule(HyperFlowDataProcessAlgorithm):
    class__ = "NeighbourhoodCleaningRule"
    module__ = "imblearn.under_sampling"
