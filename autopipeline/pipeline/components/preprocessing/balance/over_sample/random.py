from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["RandomOverSampler"]


class RandomOverSampler(AutoPLDataProcessAlgorithm):
    class__ = "RandomOverSampler"
    module__ = "imblearn.over_sampling"
