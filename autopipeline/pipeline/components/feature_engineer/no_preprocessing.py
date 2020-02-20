from autopipeline.pipeline.components.base import AutoPLPreprocessingAlgorithm


class NoPreprocessing( AutoPLPreprocessingAlgorithm):

    def fit(self,X,y=None):
        self.preprocessor=0
        return self

    def transform(self, X):
        return X


