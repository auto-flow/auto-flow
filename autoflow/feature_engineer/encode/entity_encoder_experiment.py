#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
import pandas as pd
import pylab as plt

from autoflow.feature_engineer.encode import EntityEncoder
from autoflow.utils.logging_ import setup_logger


def plot_embedding(encoder, ix=0):
    if encoder.transform_matrix_status == "Updated":
        tm = encoder.transform_matrix[ix]
        categories = encoder.ordinal_encoder.categories_[ix]
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
    mask = df['estimator'].isin(['lightgbm', 'random_forest', 'extra_trees'])
    df = df.loc[mask, :]
    y = y[mask]
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

def run_multi_cat_vars():
    df = pd.read_csv("estimator_accuracy.csv")
    df = df[['estimator','feature_engineer','loss','highc_cat_encoder','cost_time','minimum_fraction', 'accuracy']]
    y = df.pop('accuracy')
    encoder = EntityEncoder(
        cols=['estimator','feature_engineer','highc_cat_encoder','minimum_fraction'],
        max_epoch=300,
        early_stopping_rounds=100,
        update_accepted_samples=10,
        update_used_samples=100,
        budget=np.inf
    )
    encoder.fit(df, y)
    X_trans=encoder.transform(df)
    X_inv=encoder.inverse_transform(X_trans)
    print(X_inv)
    np.all(X_inv == df)
    # for i in range(4):
    #     plot_embedding(encoder,i)

def run_contain_nan():
    df = pd.read_csv("estimator_accuracy.csv")
    df = df[['estimator','feature_engineer','loss','highc_cat_encoder','cost_time','minimum_fraction', 'af_hidden', 'accuracy']]
    y = df.pop('accuracy')
    encoder = EntityEncoder(
        cols=['estimator','feature_engineer','highc_cat_encoder','minimum_fraction','af_hidden'],
        max_epoch=200,
        early_stopping_rounds=100,
        update_accepted_samples=10,
        update_used_samples=100,
        budget=np.inf,n_jobs=1
    )
    encoder.fit(df, y)
    for i in range(5):
        plot_embedding(encoder,i)
    X_trans=encoder.transform(df)
    print(X_trans)
    X_trans[pd.isna(X_trans)]=0
    X_inv=encoder.inverse_transform(X_trans)
    df2=df.copy()
    df2['af_hidden'][pd.isna(df2['af_hidden'])] = 'leaky_relu'
    print(X_inv)
    assert np.all(df2==X_inv)
    # np.all(X_inv == df)

if __name__ == '__main__':
    setup_logger()
    # run_contain_nan()
    # run_multi_cat_vars()
    run_iterative()
    # run_all()
    # run_subset()
