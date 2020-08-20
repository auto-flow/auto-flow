from autoflow.workflow.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["BorderlineSMOTE"]


class BorderlineSMOTE(AutoFlowDataProcessAlgorithm):
    class__ = "BorderlineSMOTE"
    module__ = "imblearn.over_sampling"
