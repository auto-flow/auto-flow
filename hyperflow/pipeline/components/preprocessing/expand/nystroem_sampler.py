from hyperflow.pipeline.components.feature_engineer_base import HyperFlowFeatureEngineerAlgorithm

__all__=["Nystroem"]

class Nystroem(HyperFlowFeatureEngineerAlgorithm):
    class__ = "Nystroem"
    module__ = "sklearn.kernel_approximation"