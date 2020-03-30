from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["InstanceHardnessThreshold"]


class InstanceHardnessThreshold(AutoPLDataProcessAlgorithm):
    class__ = "InstanceHardnessThreshold"
    module__ = "imblearn.under_sampling"
