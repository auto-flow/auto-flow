from autoflow.workflow.components.classification_base import AutoFlowClassificationAlgorithm

__all__ = ["BernoulliNB"]


class BernoulliNB(AutoFlowClassificationAlgorithm):
    class__ = "BernoulliNB"
    module__ = "sklearn.naive_bayes"
    OVR__ = True
