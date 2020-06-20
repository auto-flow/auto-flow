#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
from frozendict import frozendict

from autoflow.hpbandster.core.master import Master
from autoflow.hpbandster.optimizers.iterations.successivehalving import SuccessiveHalving
from autoflow.hpbandster.utils import get_max_SH_iter


class Optimizer(Master):
    def __init__(
            self,
            run_id,
            config_generator,
            working_directory='.',
            ping_interval=60,
            nameserver='127.0.0.1',
            nameserver_port=None,
            host=None,
            shutdown_workers=True,
            job_queue_sizes=(-1, 0),
            dynamic_queue_size=True,
            result_logger=None,
            previous_result=None,
            min_budget=1 / 16,
            max_budget=1,
            eta=4,
            SH_only=False,
    ):
        self.eta = eta
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.SH_only = SH_only
        self.max_SH_iter = get_max_SH_iter(self.min_budget, self.max_budget, self.eta)
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0, self.max_SH_iter))
        super(Optimizer, self).__init__(
            run_id,
            config_generator,
            working_directory,
            ping_interval,
            nameserver,
            nameserver_port,
            host,
            shutdown_workers,
            job_queue_sizes,
            dynamic_queue_size,
            result_logger,
            previous_result
        )

    def get_next_iteration(self, iteration, iteration_kwargs=frozendict()):
        """
        BO-HB uses (just like Hyperband) SuccessiveHalving for each iteration.
        See Li et al. (2016) for reference.

        Parameters
        ----------
            iteration: int
                the index of the iteration to be instantiated

        Returns
        -------
            SuccessiveHalving: the SuccessiveHalving iteration with the
                corresponding number of configurations
        """

        iteration_kwargs = dict(iteration_kwargs)
        # number of 'SH rungs'v
        if self.SH_only:
            s = self.max_SH_iter - 1
        else:
            s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]
        return SuccessiveHalving(
            HPB_iter=iteration,
            num_configs=ns,
            budgets=self.budgets[(-s - 1):],
            config_sampler=self.config_generator.get_config,
            **iteration_kwargs
        )
