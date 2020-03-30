from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm

__all__ = ["ClusterCentroids"]


class ClusterCentroids(AutoPLDataProcessAlgorithm):
    class__ = "ClusterCentroids"
    module__ = "imblearn.under_sampling"
