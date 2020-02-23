from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm

__all__=["QDA"]

class QDA(AutoPLClassificationAlgorithm):
    class__ = "QuadraticDiscriminantAnalysis"
    module__ = "sklearn.discriminant_analysis"
    OVR__ = True