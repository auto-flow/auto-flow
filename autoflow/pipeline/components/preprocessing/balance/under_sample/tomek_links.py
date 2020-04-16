from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["TomekLinks"]


class TomekLinks(AutoFlowDataProcessAlgorithm):
    class__ = "TomekLinks"
    module__ = "imblearn.under_sampling"
