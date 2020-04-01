from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm

__all__=["QDA"]

class QDA(HyperFlowClassificationAlgorithm):
    class__ = "QuadraticDiscriminantAnalysis"
    module__ = "sklearn.discriminant_analysis"
    OVR__ = True