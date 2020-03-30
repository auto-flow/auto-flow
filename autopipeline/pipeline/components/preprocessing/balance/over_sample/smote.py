from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["SMOTE"]


class SMOTE(AutoPLDataProcessAlgorithm):
    class__ = "SMOTE"
    module__ = "imblearn.over_sampling"
