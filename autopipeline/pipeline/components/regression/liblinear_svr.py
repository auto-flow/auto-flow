from autopipeline.pipeline.components.base import AutoPLRegressionAlgorithm


class LibLinear_SVR(AutoPLRegressionAlgorithm):
    class__ = "LinearSVR"
    module__ = "sklearn.svm"