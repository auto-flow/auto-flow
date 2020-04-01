from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["InstanceHardnessThreshold"]


class InstanceHardnessThreshold(HyperFlowDataProcessAlgorithm):
    class__ = "InstanceHardnessThreshold"
    module__ = "imblearn.under_sampling"
