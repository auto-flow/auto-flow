from autopipeline.pipeline.components.base import AutoPLClassificationAlgorithm

class MultinomialNB(AutoPLClassificationAlgorithm):
    module__ = "sklearn.naive_bayes"
    class__ = "MultinomialNB"
    OVR__ = True

