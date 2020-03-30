from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["AllKNN"]


class AllKNN(AutoPLDataProcessAlgorithm):
    class__ = "AllKNN"
    module__ = "imblearn.under_sampling"
