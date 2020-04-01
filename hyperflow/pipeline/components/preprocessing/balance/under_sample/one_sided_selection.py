from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["OneSidedSelection"]


class OneSidedSelection(HyperFlowDataProcessAlgorithm):
    class__ = "OneSidedSelection"
    module__ = "imblearn.under_sampling"
