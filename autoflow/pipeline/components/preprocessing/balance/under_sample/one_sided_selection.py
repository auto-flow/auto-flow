from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["OneSidedSelection"]


class OneSidedSelection(AutoFlowDataProcessAlgorithm):
    class__ = "OneSidedSelection"
    module__ = "imblearn.under_sampling"
