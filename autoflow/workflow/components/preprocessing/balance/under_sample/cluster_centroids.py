from autoflow.workflow.components.data_process_base import AutoFlowDataProcessAlgorithm

__all__ = ["ClusterCentroids"]


class ClusterCentroids(AutoFlowDataProcessAlgorithm):
    class__ = "ClusterCentroids"
    module__ = "imblearn.under_sampling"
