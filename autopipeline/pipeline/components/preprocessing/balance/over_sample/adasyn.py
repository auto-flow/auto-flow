from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["ADASYN"]


class ADASYN(AutoPLDataProcessAlgorithm):
    class__ = "ADASYN"
    module__ = "imblearn.over_sampling"
