from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["SMOTENC"]


class SMOTENC(AutoPLDataProcessAlgorithm):
    class__ = "SMOTENC"
    module__ = "imblearn.over_sampling"
