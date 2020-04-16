from functools import partial
from typing import Dict

import pandas as pd
from imblearn import FunctionSampler
from autoflow.pipeline.components.data_process_base import AutoFlowDataProcessAlgorithm
from autoflow.pipeline.dataframe import GenericDataFrame

__all__ = ["DeleteNanRow"]

class DeleteNanRow(AutoFlowDataProcessAlgorithm):
    class__ = "DeleteNanRow"
    module__ = "autoflow.data_process.impute.delete_nan"
