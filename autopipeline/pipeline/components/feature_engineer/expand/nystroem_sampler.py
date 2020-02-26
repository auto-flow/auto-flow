from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__=["Nystroem"]

class Nystroem(AutoPLPreprocessingAlgorithm):
    class__ = "Nystroem"
    module__ = "sklearn.kernel_approximation"