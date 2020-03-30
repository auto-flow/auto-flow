from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["RandomUnderSampler"]


class RandomUnderSampler(AutoPLDataProcessAlgorithm):
    class__ = "RandomUnderSampler"
    module__ = "imblearn.under_sampling"
