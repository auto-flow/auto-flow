#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow import ResourceManager
import autoflow.hpbandster.visualization as hpvis
import pylab as plt


rm=ResourceManager()
result,_,_=rm.get_result_from_trial_table(
    task_id="2435e32babd7d09b6357e99aa7fa3b89",
    hdl_id="f289af8e23544a108bba6c8bc99673c3",
    user_id=0,
    budget_id="410cc77e823f11113afae5e323429d91",
)


# get all executed runs
all_runs = result.get_all_runs()

# get the 'dict' that translates config ids to the actual configurations
id2conf = result.get_id2config_mapping()


# Here is how you get he incumbent (best configuration)
inc_id = result.get_incumbent_id()

# let's grab the run on the highest budget
inc_runs = result.get_runs_by_id(inc_id)
inc_run = inc_runs[-1]


# We have access to all information: the config, the loss observed during
#optimization, and all the additional information
inc_loss = inc_run.loss
inc_config = id2conf[inc_id]['config']
# inc_test_loss = inc_run.info['test accuracy']

print('Best found configuration:')
print(inc_config)
print('It achieved accuracies of %f (validation) .'%(1-inc_loss))


# Let's plot the observed losses grouped by budget,
hpvis.losses_over_time(all_runs)

# the number of concurent runs,
hpvis.concurrent_runs_over_time(all_runs)

# and the number of finished runs.
hpvis.finished_runs_over_time(all_runs)

# This one visualizes the spearman rank correlation coefficients of the losses
# between different budgets.
hpvis.correlation_across_budgets(result)

# For model based optimizers, one might wonder how much the model actually helped.
# The next plot compares the performance of configs picked by the model vs. random ones
hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

plt.show()
