from autoflow.pipeline.components.classification_base import AutoFlowClassificationAlgorithm

__all__=["MultinomialNB"]

class MultinomialNB(AutoFlowClassificationAlgorithm):
    module__ = "sklearn.naive_bayes"
    class__ = "MultinomialNB"
    OVR__ = True

