from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["NearMiss"]


class NearMiss(AutoPLDataProcessAlgorithm):
    class__ = "NearMiss"
    module__ = "imblearn.under_sampling"
