from functools import partial
from typing import Dict

import pandas as pd
from imblearn import FunctionSampler
from autopipeline.pipeline.components.data_process_base import AutoPLDataProcessAlgorithm
from autopipeline.pipeline.dataframe import GenericDataFrame

__all__ = ["DeleteNanRow"]

class DeleteNanRow(AutoPLDataProcessAlgorithm):
    class__ = "DeleteNanRow"
    module__ = "autopipeline.data_process.impute.delete_nan"
