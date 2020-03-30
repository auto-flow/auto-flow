from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["TomekLinks"]


class TomekLinks(AutoPLDataProcessAlgorithm):
    class__ = "TomekLinks"
    module__ = "imblearn.under_sampling"
