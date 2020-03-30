from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["KMeansSMOTE"]


class KMeansSMOTE(AutoPLDataProcessAlgorithm):
    class__ = "KMeansSMOTE"
    module__ = "imblearn.over_sampling"
