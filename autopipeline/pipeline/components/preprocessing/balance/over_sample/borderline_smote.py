from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["BorderlineSMOTE"]


class BorderlineSMOTE(AutoPLDataProcessAlgorithm):
    class__ = "BorderlineSMOTE"
    module__ = "imblearn.over_sampling"
