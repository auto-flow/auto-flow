from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["BorderlineSMOTE"]


class BorderlineSMOTE(HyperFlowDataProcessAlgorithm):
    class__ = "BorderlineSMOTE"
    module__ = "imblearn.over_sampling"
