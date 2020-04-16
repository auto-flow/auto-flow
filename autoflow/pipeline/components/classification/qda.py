from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["QDA"]

class QDA(AutoFlowClassificationAlgorithm):
    class__ = "QuadraticDiscriminantAnalysis"
    module__ = "sklearn.discriminant_analysis"
    OVR__ = True