from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["InstanceHardnessThreshold"]


class InstanceHardnessThreshold(AutoFlowDataProcessAlgorithm):
    class__ = "InstanceHardnessThreshold"
    module__ = "imblearn.under_sampling"
