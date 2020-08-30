#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
import pandas as pd
import pylab as plt
from autoflow.feature_engineer.encode import EntityEncoder
from autoflow.utils.logging_ import setup_logger

def plot_embedding(encoder):
    if encoder.transform_matrix is not None:
        tm = encoder.transform_matrix[0]
        categories = encoder.ordinal_encoder.categories_[0]
        for i, category in enumerate(categories):
            plt.scatter(tm[i, 0], tm[i, 1], label=category)
        plt.legend(loc='best')
        plt.title(encoder.stage)
        plt.show()

def run_iterative():
    df = pd.read_csv("estimator_accuracy.csv")
    df = df[['estimator', 'accuracy']]
    y = df.pop('accuracy')
    rng = np.random.RandomState(42)
    indexes = rng.choice(np.arange(df.shape[0]), 30, False)
    encoder = EntityEncoder(
        cols=['estimator'],
        max_epoch=100,
        update_accepted_samples=10,
        update_used_samples=100,
        budget=np.inf
    )
    encoder.fit(df.loc[indexes, :], y[indexes])
    plot_embedding(encoder)
    while indexes.size < df.shape[0]:
        diff_indexes = np.setdiff1d(np.arange(df.shape[0]), indexes)
        sampled_indexes = rng.choice(diff_indexes, 1, False)
        indexes = np.hstack([indexes, sampled_indexes])
        encoder.fit(df.loc[sampled_indexes, :], y[sampled_indexes])
        plot_embedding(encoder)

def run_all():
    df = pd.read_csv("estimator_accuracy.csv")
    df = df[['estimator', 'accuracy']]
    y = df.pop('accuracy')
    encoder = EntityEncoder(
        cols=['estimator'],
        max_epoch=200,
        early_stopping_rounds=200,
        update_accepted_samples=10,
        update_used_samples=100,
        budget=np.inf
    )
    encoder.fit(df, y)
    plot_embedding(encoder)

def run_subset():
    df = pd.read_csv("estimator_accuracy.csv")
    df = df[['estimator', 'accuracy']]
    y = df.pop('accuracy')
    mask=df['estimator'].isin(['lightgbm','random_forest','extra_trees'])
    df=df.loc[mask,:]
    y=y[mask]
    encoder = EntityEncoder(
        cols=['estimator'],
        max_epoch=200,
        early_stopping_rounds=200,
        update_accepted_samples=10,
        update_used_samples=100,
        budget=np.inf
    )
    encoder.fit(df, y)
    plot_embedding(encoder)

if __name__ == '__main__':
    setup_logger()
    # run_all()
    run_subset()


