from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class Nystroem(AutoPLPreprocessingAlgorithm):
    class__ = "Nystroem"
    module__ = "sklearn.kernel_approximation"