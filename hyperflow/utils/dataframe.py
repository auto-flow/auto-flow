from typing import Optional
import numpy as np
import pandas as pd

def pop_if_exists(df:pd.DataFrame,col:str)->Optional[pd.DataFrame]:
    if col in df.columns:
        return df.pop(col)
    else:
        return None

if __name__ == '__main__':
    df=pd.DataFrame(np.arange(9).reshape([3,3]),columns=["a","b","c"])
    result=pop_if_exists(df,"a")
    print(result)
    print(df)