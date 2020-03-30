from autopipeline.pipeline.components.feature_engineer_base import AutoPLFeatureEngineerAlgorithm

__all__=["Nystroem"]

class Nystroem(AutoPLFeatureEngineerAlgorithm):
    class__ = "Nystroem"
    module__ = "sklearn.kernel_approximation"