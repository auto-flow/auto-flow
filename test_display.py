#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

from IPython.display import display,HTML

import pandas as pd
import numpy as np

def do_display():
    df1=pd.DataFrame(np.zeros([3, 4]))
    print("Dataframe 1:")
    display(df1)
    print("Dataframe 2:")
    display(HTML(df1.to_html()))
