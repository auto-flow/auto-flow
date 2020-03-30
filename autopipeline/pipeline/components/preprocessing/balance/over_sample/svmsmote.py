from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["SVMSMOTE"]


class SVMSMOTE(AutoPLDataProcessAlgorithm):
    class__ = "SVMSMOTE"
    module__ = "imblearn.over_sampling"
