#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pylab as plt


def raw2min(df: pd.DataFrame):
    df_m = pd.DataFrame(np.zeros_like(df.values), columns=df.columns)
    for i in range(df.shape[0]):
        df_m.loc[i, :] = df.loc[:i, :].min()
    return df_m


# ix2info = json.loads(Path("plot_configs/400_comparison.json").read_text())
# ix2info = json.loads(Path("plot_configs/1000_comparison.json").read_text())
# ix2info = json.loads(Path("plot_configs/ambo_comparison.json").read_text())
ix2info = json.loads(Path("plot_configs/tpe_comparison.json").read_text())
for ix, (title, color) in ix2info.items():
    df_m = raw2min(pd.read_csv(f"{ix}.csv"))
    # df_m = df_m.iloc[:400, :]
    print(df_m)
    plt.grid()
    mean = df_m.mean(1)
    std = df_m.std(1)
    iters = range(df_m.shape[0])
    plt.grid()
    plt.fill_between(
        iters, mean - std, mean + std, alpha=0.1,
        color=color
    )

    plt.plot(
        iters, mean, color=color, label=title, alpha=0.9
    )
plt.grid(alpha=0.4)
plt.legend(loc="best")
plt.xlabel("iterations")
plt.ylabel("losses")
plt.title("Comparison of BO algorithms")
plt.savefig("compare.png")
plt.show()
