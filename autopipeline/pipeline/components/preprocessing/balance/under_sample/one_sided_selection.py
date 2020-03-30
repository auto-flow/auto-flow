from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["OneSidedSelection"]


class OneSidedSelection(AutoPLDataProcessAlgorithm):
    class__ = "OneSidedSelection"
    module__ = "imblearn.under_sampling"
