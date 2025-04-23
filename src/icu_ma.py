from typing import Union

import numpy as np
import pandas as pd


def siegelslopes_ma(price_ser: Union[pd.Series, np.ndarray],method:str="hierarchical") -> float:
    """Repeated Median (Siegel 1982)

    Args:
        price_ser (Union[pd.Series, np.ndarray]): index-date values-price or values-price

    Returns:
        float: float
    """
    from scipy import stats
    n: int = len(price_ser)
    res = stats.siegelslopes(price_ser, np.arange(n), method=method)
    return res.intercept + res.slope * (n-1)


def calc_icu_ma(price:pd.Series,N:int)->pd.Series:
    """计算ICU均线

    Args:
        price (pd.Series): index-date values-price
        N (int): 计算窗口
    Returns:
        pd.Series: index-date values-icu_ma
    """
    if len(price) <= N:
        raise ValueError("price length must be greater than N")
    
    return price.rolling(N).apply(siegelslopes_ma,raw=True)