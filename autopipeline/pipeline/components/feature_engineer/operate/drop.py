from autopipeline.pipeline.components.preprocess_base import AutoPLPreprocessingAlgorithm

__all__ = ["DropAll"]


class DropAll(AutoPLPreprocessingAlgorithm):
    class__ = "DropAll"
    module__ = "autopipeline.pipeline.proxy.drop_all"