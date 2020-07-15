#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from typing import List

import numpy as np
from ConfigSpace import Configuration
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity

from autoflow.opt.config_generators.density_estimator.base import BaseDensityEstimator
from autoflow.opt.utils import LabelTsneEncoder, MixedTsneEncoder


def estimate_bw(data, bw_method="scott", cv_times=100):
    ndata = data.shape[0]
    if bw_method == 'scott':
        bandwidth = ndata ** (-1 / 5) * np.std(data, ddof=1)
        bandwidth = np.clip(bandwidth, 0.01, None)
    elif bw_method == 'silverman':
        bandwidth = (ndata * 3 / 4) ** (-1 / 5) * np.std(data, ddof=1)
        bandwidth = np.clip(bandwidth, 0.01, None)
    elif bw_method == 'cv':
        if ndata <= 3:
            return estimate_bw(data)
        bandwidths = np.std(data, ddof=1) ** np.linspace(-1, 1, cv_times)
        bandwidths = np.clip(bandwidths, 0.01, None)
        grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths},
                            cv=KFold(n_splits=3, shuffle=True, random_state=0))
        grid.fit(data)
        bandwidth = grid.best_params_['bandwidth']
    elif np.isscalar(bw_method):
        bandwidth = bw_method
    else:
        raise ValueError("Unrecognized input for bw_method.")
    return bandwidth


class TreeStructuredParzenEstimator(BaseDensityEstimator):
    def __init__(
            self,
            top_n_percent=15, bandwidth_factor=3, min_bandwidth=1e3,
            bw_estimation="normal_reference", min_points_in_kde=2,
            bw_method="scott", cv_times=100, kde_sample_weight_scaler=None,

            fill_deactivated_value=False
    ):
        super(TreeStructuredParzenEstimator, self).__init__(
            top_n_percent, bandwidth_factor, min_bandwidth,
            bw_estimation, min_points_in_kde
        )
        self.kde_sample_weight_scaler = kde_sample_weight_scaler
        self.cv_times = cv_times
        self.bw_method = bw_method
        self.fill_deactivated_value = fill_deactivated_value
        self.good_kdes = None
        self.bad_kdes = None

    def set_config_transformer(self, config_transformer):
        self.config_transformer = config_transformer
        n_choices_list = np.array(self.config_transformer.n_choices_list)
        n_groups = self.config_transformer.n_groups
        mixed_tsne_encoders = np.zeros([n_groups], dtype=object)
        label_tsne_encoders = np.zeros([len(n_choices_list)], dtype=object)
        groups = np.array(self.config_transformer.groups)
        for i, n_choices in enumerate(n_choices_list):
            if n_choices > 2:
                label_tsne_encoders[i] = LabelTsneEncoder(n_choices)
        for group in range(n_groups):
            group_mask = groups == group
            grouped_label_tsne_encoders = label_tsne_encoders[group_mask]
            mixed_tsne_encoder = MixedTsneEncoder(grouped_label_tsne_encoders, n_choices_list[group_mask])
            mixed_tsne_encoders[group] = mixed_tsne_encoder
        self.mixed_tsne_encoders = mixed_tsne_encoders

    def fit(self, X: np.ndarray, y: np.ndarray):
        super(TreeStructuredParzenEstimator, self).fit(X, y)
        # 对n_choices>2的列，没列构造一个label_tsne_encoder
        n_groups = self.config_transformer.n_groups
        good_kdes = np.zeros([n_groups], dtype=object)
        bad_kdes = deepcopy(good_kdes)

        groups = np.array(self.config_transformer.groups)
        for group in range(n_groups):
            group_mask = groups == group
            grouped_X = X[:, group_mask]
            inactive_mask = np.isnan(grouped_X[:, 0])
            active_X = grouped_X[~inactive_mask, :]
            active_y = y[~inactive_mask]
            if active_X.shape[0] < 4:  # at least have 4 samples
                continue
            N, M = active_X.shape
            # Each KDE contains at least 2 samples
            n_good = max(2, (self.top_n_percent * N) // 100)
            if n_good < self.min_points_in_kde or \
                    N - n_good < self.min_points_in_kde:
                # Too few observation samples
                continue
            mixed_tsne_encoder = self.mixed_tsne_encoders[group]
            transformed_X = mixed_tsne_encoder.transform(active_X)
            idx = np.argsort(active_y)
            X_good = transformed_X[idx[:n_good]]
            X_bad = transformed_X[idx[n_good:]]
            y_good = -active_y[idx[:n_good]]
            sample_weight = None
            if self.kde_sample_weight_scaler is not None and y_good.std() != 0:
                if self.kde_sample_weight_scaler == "normalize":
                    scaled_y = (y_good - y_good.mean()) / y_good.std()
                    scaled_y -= np.min(scaled_y)
                    scaled_y /= np.max(scaled_y)
                    scaled_y += 0.5
                    sample_weight = scaled_y
                elif self.kde_sample_weight_scaler == "std-exp":
                    scaled_y = (y_good - y_good.mean()) / y_good.std()
                    sample_weight = np.exp(scaled_y)
                else:
                    raise ValueError(f"Invalid kde_sample_weight_scaler '{self.kde_sample_weight_scaler}'")
            bw_good = estimate_bw(X_good, self.bw_method, self.cv_times)
            bw_bad = estimate_bw(X_bad, self.bw_method, self.cv_times)
            good_kdes[group] = KernelDensity(bw_good).fit(X_good, sample_weight=sample_weight)
            bad_kdes[group] = KernelDensity(bw_bad).fit(X_bad)
        self.good_kdes = good_kdes
        self.bad_kdes = bad_kdes
        return self

    def predict(self, X: np.ndarray):
        super(TreeStructuredParzenEstimator, self).predict(X)
        n_groups = self.config_transformer.n_groups
        good_log_pdf = np.zeros([X.shape[0], n_groups], dtype="float64")
        bad_log_pdf = deepcopy(good_log_pdf)
        groups = np.array(self.config_transformer.groups)
        for group, (good_kde, bad_kde) in enumerate(zip(self.good_kdes, self.bad_kdes)):
            if (not good_kde) or (not bad_kde):
                continue
            group_mask = groups == group
            grouped_X = X[:, group_mask]
            inactive_mask = np.isnan(grouped_X[:, 0])
            active_X = grouped_X[~inactive_mask, :]
            N, M = active_X.shape
            if N == 0:
                continue
            mixed_tsne_encoder = self.mixed_tsne_encoders[group]
            transformed_X = mixed_tsne_encoder.transform(active_X)
            good_log_pdf[~inactive_mask, group] = self.good_kdes[group].score_samples(transformed_X)
            bad_log_pdf[~inactive_mask, group] = self.bad_kdes[group].score_samples(transformed_X)
            # if N_deactivated > 0 and self.fill_deactivated_value:
            #     good_log_pdf[~mask, i] = np.random.choice(good_pdf_activated)
            #     bad_log_pdf[~mask, i] = np.random.choice(bad_pdf_activated)
        if not np.all(np.isfinite(good_log_pdf)):
            self.logger.warning("good_log_pdf contains NaN or inf")
        if not np.all(np.isfinite(bad_log_pdf)):
            self.logger.warning("bad_log_pdf contains NaN or inf")
        good_log_pdf[~np.isfinite(good_log_pdf)] = -10
        bad_log_pdf[bad_log_pdf == -np.inf] = -10
        bad_log_pdf[~np.isfinite(bad_log_pdf)] = 10
        result = good_log_pdf.sum(axis=1) - bad_log_pdf.sum(axis=1)
        return result

    def sample(self, n_candidates=20, sort_by_EI=False) -> List[Configuration]:
        n_choices_list = np.array(self.config_transformer.n_choices_list)
        groups = np.array(self.config_transformer.groups)
        # n_groups = self.config_transformer.n_groups
        if self.good_kdes is None:
            self.logger.warning("good_kdes is None, random sampling.")
            return self.config_transformer.config_space.sample_configuration(n_candidates)
        sampled_matrix = np.zeros([n_candidates, len(n_choices_list)])
        for group, (good_kde, mixed_tsne_encoder) in enumerate(zip(self.good_kdes, self.mixed_tsne_encoders)):
            group_mask = groups == group
            if good_kde:
                # KDE采样
                result = good_kde.sample(n_candidates)
                result = mixed_tsne_encoder.inverse_transform(result)
            else:
                # 随机采样
                n_choices = n_choices_list[group_mask]
                result = np.zeros([n_candidates, 0])
                for n_choice in n_choices:
                    if n_choice == 0:
                        col = np.random.rand(n_candidates)
                    else:
                        col = np.random.randint(0, n_choice, [n_candidates])
                    result = np.hstack([result, col[:, None]])
            sampled_matrix[:, group_mask] = result
        if sort_by_EI:
            # fixme: don't care conditional variable
            EI = self.predict(sampled_matrix)
            idx = np.argsort(-EI)
            sampled_matrix = sampled_matrix[idx, :]
        candidates = []
        n_fails = 0
        for i in range(n_candidates):
            config = self.config_transformer.inverse_transform(sampled_matrix[i, :])
            if config is not None:
                candidates.append(config)
            else:
                n_fails += 1
        if n_fails:
            candidates.append(self.config_transformer.config_space.sample_configuration(n_fails))
        return candidates
