from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["TomekLinks"]


class TomekLinks(HyperFlowDataProcessAlgorithm):
    class__ = "TomekLinks"
    module__ = "imblearn.under_sampling"
