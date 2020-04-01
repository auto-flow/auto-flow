from hyperflow.pipeline.components.classification_base import HyperFlowClassificationAlgorithm

__all__=["MultinomialNB"]

class MultinomialNB(HyperFlowClassificationAlgorithm):
    module__ = "sklearn.naive_bayes"
    class__ = "MultinomialNB"
    OVR__ = True

