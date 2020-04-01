from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm

__all__ = ["ClusterCentroids"]


class ClusterCentroids(HyperFlowDataProcessAlgorithm):
    class__ = "ClusterCentroids"
    module__ = "imblearn.under_sampling"
