from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__ = ["QuadraticDiscriminantAnalysis"]


class QuadraticDiscriminantAnalysis(AutoFlowClassificationAlgorithm):
    class__ = "QuadraticDiscriminantAnalysis"
    module__ = "sklearn.discriminant_analysis"
    OVR__ = True
