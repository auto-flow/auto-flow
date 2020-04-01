from functools import partial
from typing import Dict

import pandas as pd
from imblearn import FunctionSampler
from hyperflow.pipeline.components.data_process_base import HyperFlowDataProcessAlgorithm
from hyperflow.pipeline.dataframe import GenericDataFrame

__all__ = ["DeleteNanRow"]

class DeleteNanRow(HyperFlowDataProcessAlgorithm):
    class__ = "DeleteNanRow"
    module__ = "hyperflow.data_process.impute.delete_nan"
