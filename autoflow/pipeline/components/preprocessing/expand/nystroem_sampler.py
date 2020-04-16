from autoflow.pipeline.components.feature_engineer_base import AutoFlowFeatureEngineerAlgorithm

__all__=["Nystroem"]

class Nystroem(AutoFlowFeatureEngineerAlgorithm):
    class__ = "Nystroem"
    module__ = "sklearn.kernel_approximation"