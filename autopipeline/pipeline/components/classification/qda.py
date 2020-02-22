from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm


class QDA(AutoPLClassificationAlgorithm):
    class__ = "QuadraticDiscriminantAnalysis"
    module__ = "sklearn.discriminant_analysis"
    OVR__ = True