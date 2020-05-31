import logging
import pickle
import time
import typing
from collections import Counter
from typing import Callable

import numpy as np

from dsmac.configspace import Configuration
from dsmac.optimizer.ei_optimization import ChallengerList
from dsmac.optimizer.objective import sum_cost
from dsmac.runhistory.runhistory import RunHistory
from dsmac.runhistory.structure import RunKey
from dsmac.runhistory.utils import get_id_of_config
from dsmac.stats.stats import Stats
from dsmac.tae.execute_ta_run import BudgetExhaustedException, CappedRunException, ExecuteTARun
from dsmac.utils.constants import MAXINT, MAX_CUTOFF
from dsmac.utils.io.traj_logging import TrajLogger

__author__ = "Katharina Eggensperger, Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class Intensifier(object):
    """Races challengers against an incumbent (a.k.a. SMAC's intensification
    procedure).


    Parameters
    ----------
    tae_runner : tae.executre_ta_run_*.ExecuteTARun* Object
        target algorithm run executor
    stats: Stats
        stats object
    traj_logger: TrajLogger
        TrajLogger object to log all new incumbents
    rng : np.random.RandomState
    instances : typing.List[str]
        list of all instance ids
    instance_specifics : typing.Mapping[str,np.ndarray]
        mapping from instance name to instance specific string
    cutoff : int
        runtime cutoff of TA runs
    deterministic: bool
        whether the TA is deterministic or not
    run_obj_time: bool
        whether the run objective is runtime or not (if true, apply adaptive capping)
    always_race_against: Configuration
        if incumbent changes race this configuration always against new incumbent;
        can sometimes prevent over-tuning
    use_ta_time_bound: bool,
        if true, trust time reported by the target algorithms instead of
        measuring the wallclock time for limiting the time of intensification
    run_limit : int
        Maximum number of target algorithm runs per call to intensify.
    maxR : int
        Maximum number of runs per config (summed over all calls to
        intensifiy).
    minR : int
        Minimum number of run per config (summed over all calls to
        intensify).
    adaptive_capping_slackfactor: float
        slack factor of adpative capping (factor * adpative cutoff)
    min_chall: int
        minimal number of challengers to be considered
        (even if time_bound is exhausted earlier)
    """

    def __init__(self, tae_runner: ExecuteTARun, stats: Stats,
                 traj_logger: TrajLogger, rng: np.random.RandomState,
                 instances: typing.List[str],
                 instance_specifics: typing.Mapping[str, np.ndarray] = None,
                 cutoff: int = MAX_CUTOFF, deterministic: bool = False,
                 run_obj_time: bool = True,
                 always_race_against: Configuration = None,
                 run_limit: int = MAXINT,
                 use_ta_time_bound: bool = False,
                 minR: int = 1, maxR: int = 2000,
                 adaptive_capping_slackfactor: float = 1.2,
                 min_chall: int = 2,
                 filter_callback: Callable = lambda x: True):
        self.filter_callback = filter_callback
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.stats = stats
        self.traj_logger = traj_logger
        # general attributes
        if instances is None:
            instances = []
        self.instances = set(instances)

        self.instance = instances[0] if len(instances) > 0 else None
        if instance_specifics is None:
            self.instance_specifics = {}
        else:
            self.instance_specifics = instance_specifics
        self.run_limit = run_limit
        self.maxR = maxR
        self.minR = minR
        self.rs = rng

        self.always_race_against = always_race_against

        # scenario info
        self.cutoff = cutoff
        self.deterministic = deterministic
        self.run_obj_time = run_obj_time
        self.tae_runner = tae_runner

        self.adaptive_capping_slackfactor = adaptive_capping_slackfactor

        if self.run_limit < 1:
            raise ValueError("run_limit must be > 1")

        self._num_run = 0
        self._chall_indx = 0

        self._ta_time = 0
        self.use_ta_time_bound = use_ta_time_bound
        self._min_time = 10 ** -5
        self.min_chall = min_chall

    def intensify(self, challengers: typing.List[Configuration],
                  incumbent: Configuration,
                  run_history: RunHistory,
                  aggregate_func: typing.Callable,
                  time_bound: float = float(MAXINT),
                  log_traj: bool = True,
                  anneal_func=None):
        """Running intensification to determine the incumbent configuration.
        *Side effect:* adds runs to run_history

        Implementation of Procedure 2 in Hutter et al. (2011).

        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        incumbent : Configuration
            best configuration so far
        run_history : RunHistory
            stores all runs we ran so far
        aggregate_func: typing.Callable
            aggregate error across instances
        time_bound : float, optional (default=2 ** 31 - 1)
            time in [sec] available to perform intensify
        log_traj: bool
            whether to log changes of incumbents in trajectory

        Returns
        -------
        incumbent: Configuration()
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """
        self.start_time = time.time()
        self._ta_time = 0

        if time_bound < self._min_time:
            raise ValueError("time_bound must be >= %f" % (self._min_time))

        self._num_run = 0
        self._chall_indx = 0

        # Line 1 + 2
        if isinstance(challengers, ChallengerList):
            challengers = challengers.challengers

        if anneal_func is not None:
            # todo: 控制随机种子？
            r = np.random.uniform()
            try:
                target = int(round(anneal_func(r)))
            except Exception as e:
                self.logger.error(f"Error occurred in calculate anneal_func({r})\n{e}")
                target = 0
            target = min(target, len(challengers) - 1)
            self.logger.debug(f"By anneal_func , target is {target}.")
            challengers[target], challengers[0] = challengers[0], challengers[target]

        for challenger in challengers:
            appoint_success, record = run_history.db.appointment_config(challenger, self.instance)
            if not appoint_success:
                if record is not None:
                    cost = record.cost
                    config = record.config_bin
                    if isinstance(config,bytes):
                        config=pickle.loads(config)
                    run_history.add(config, cost, record.time, record.status,
                                    record.instance_id)
                    if cost < run_history.get_cost(incumbent):
                        incumbent = config
                continue
            if challenger == incumbent:
                self.logger.debug("Challenger was the same as the current incumbent; Skipping challenger")
                continue
            # fixme: 对challenger是否存在runhistory做一个判断
            run_key = RunKey(get_id_of_config(challenger), self.instance, 0)
            if run_key in run_history.data:
                self.logger.debug(f"run_key = {run_key} is already existing in run_history, pass.")
                continue
            self.logger.debug("Intensify on %s", challenger)
            if hasattr(challenger, 'origin'):
                self.logger.debug(
                    "Configuration origin: %s", challenger.origin)

            try:
                # Lines 3-7
                self._add_inc_run(incumbent=incumbent, run_history=run_history)

                # Lines 8-17
                incumbent = self._race_challenger(challenger=challenger,
                                                  incumbent=incumbent,
                                                  run_history=run_history,
                                                  aggregate_func=aggregate_func,
                                                  log_traj=log_traj)

            except BudgetExhaustedException:
                # We return incumbent, SMBO stops due to its own budget checks
                inc_perf = run_history.get_cost(incumbent)
                self.logger.debug("Budget exhausted; Return incumbent")
                return incumbent, inc_perf

            tm = time.time()
            if self._chall_indx >= self.min_chall:
                if self._num_run > self.run_limit:
                    self.logger.debug(
                        "Maximum #runs for intensification reached")
                    break
                # 一般来说在这里退出。(tm - self.start_time) 表示程序运行时间, 如果大于 time_bound(1e5)说明程序进行了
                # 运算，那么就退出。
                if not self.use_ta_time_bound and tm - self.start_time - time_bound >= 0:
                    self.logger.debug("Wallclock time limit for intensification reached ("
                                      "used: %f sec, available: %f sec)" %
                                      (tm - self.start_time, time_bound))
                    break
                elif self._ta_time - time_bound >= 0:
                    self.logger.debug("TA time limit for intensification reached ("
                                      "used: %f sec, available: %f sec)" %
                                      (self._ta_time, time_bound))
                    break

        # output estimated performance of incumbent
        inc_runs = run_history.get_runs_for_config(incumbent)
        inc_perf = aggregate_func(incumbent, run_history, inc_runs)
        self.logger.info("Updated estimated cost of incumbent on %d runs: %.4f"
                         % (len(inc_runs), inc_perf))

        self.stats.update_average_configs_per_intensify(
            n_configs=self._chall_indx)

        return incumbent, inc_perf

    def _add_inc_run(self, incumbent: Configuration, run_history: RunHistory):
        """Add new run for incumbent

        *Side effect:* adds runs to <run_history>

        Parameters
        ----------
        incumbent : Configuration
            best configuration so far
        run_history : RunHistory
            stores all runs we ran so far
        """
        inc_runs = run_history.get_runs_for_config(incumbent)

        # Line 3
        # First evaluate incumbent on a new instance
        if len(inc_runs) < self.maxR:
            while True:
                # Line 4
                # find all instances that have the most runs on the inc
                inc_runs = run_history.get_runs_for_config(incumbent)
                inc_inst = [s.instance for s in inc_runs]
                inc_inst = list(Counter(inc_inst).items())
                inc_inst.sort(key=lambda x: x[1], reverse=True)
                try:
                    max_runs = inc_inst[0][1]
                except IndexError:
                    self.logger.debug("No run for incumbent found")
                    max_runs = 0
                inc_inst = set([x[0] for x in inc_inst if x[1] == max_runs])

                available_insts = (self.instances - inc_inst)

                # if all instances were used n times, we can pick an instances
                # from the complete set again
                if not self.deterministic and not available_insts:
                    available_insts = self.instances

                # Line 6 (Line 5 is further down...)
                if self.deterministic:
                    next_seed = 0
                else:
                    next_seed = self.rs.randint(low=0, high=MAXINT,
                                                size=1)[0]

                if available_insts:
                    # Line 5 (here for easier code)
                    next_instance = self.rs.choice(list(available_insts))
                    # Line 7
                    self.logger.debug("Add run of incumbent")
                    status, cost, dur, result = self.tae_runner.start(
                        config=incumbent,
                        instance=next_instance,
                        seed=next_seed,
                        cutoff=self.cutoff,
                        instance_specific=self.instance_specifics.get(next_instance, "0"))
                    self._ta_time += dur
                    self._num_run += 1
                else:
                    self.logger.debug("No further instance-seed pairs for "
                                      "incumbent available.")
                    break

                inc_runs = run_history.get_runs_for_config(incumbent)
                # Termination condition; after exactly one run, this checks
                # whether further runs are necessary due to minR
                if len(inc_runs) >= self.minR or len(inc_runs) >= self.maxR:
                    break

    def _race_challenger(self, challenger: Configuration,
                         incumbent: Configuration,
                         run_history: RunHistory,
                         aggregate_func: typing.Callable,
                         log_traj: bool = True):
        """Aggressively race challenger against incumbent

        Parameters
        ----------
        challenger : Configuration
            Configuration which challenges incumbent
        incumbent : Configuration
            Best configuration so far
        run_history : RunHistory
            Stores all runs we ran so far
        aggregate_func: typing.Callable
            Aggregate performance across instances
        log_traj: bool
            Whether to log changes of incumbents in trajectory

        Returns
        -------
        new_incumbent: Configuration
            Either challenger or incumbent
        """
        # at least one run of challenger
        # to increase chall_indx counter
        first_run = False

        # Line 8
        N = max(1, self.minR)

        inc_inst_seeds = set(run_history.get_runs_for_config(incumbent))
        # Line 9
        while True:
            # cost used by incumbent for going over all runs in inst_seed_pairs
            inc_sum_cost = sum_cost(config=incumbent,
                                    instance_seed_pairs=None,
                                    run_history=run_history)

            cutoff = self._adapt_cutoff(challenger=challenger,
                                        incumbent=incumbent,
                                        run_history=run_history,
                                        inc_sum_cost=inc_sum_cost)
            if cutoff is not None and cutoff <= 0:
                # no time to validate challenger
                self.logger.debug("Stop challenger itensification due "
                                  "to adaptive capping.")
                # challenger performance is worse than incumbent
                return incumbent

            if not first_run:
                first_run = True
                self._chall_indx += 1

            self.logger.debug("Add run of challenger")
            # fixme: 我删除了这里一个意义不明的for循环，并且修改了instance等传入参数
            try:
                status, cost, dur, result = self.tae_runner.start(
                    config=challenger,
                    instance=self.instance,
                    seed=0,
                    cutoff=cutoff,
                    instance_specific=self.instance_specifics.get(
                        self.instance, "0"),
                    capped=(self.cutoff is not None) and
                           (cutoff < self.cutoff))
                self._num_run += 1
                self._ta_time += dur
            except CappedRunException:
                return incumbent

            new_incumbent = self._compare_configs(
                incumbent=incumbent, challenger=challenger,
                run_history=run_history,
                aggregate_func=aggregate_func,
                log_traj=log_traj)
            if new_incumbent == incumbent:
                break
            elif new_incumbent == challenger:
                incumbent = challenger
                break
            else:  # Line 17
                # challenger is not worse, continue
                N = 2 * N

        return incumbent

    def _adapt_cutoff(self, challenger: Configuration,
                      incumbent: Configuration,
                      run_history: RunHistory,
                      inc_sum_cost: float):
        """Adaptive capping:
        Compute cutoff based on time so far used for incumbent
        and reduce cutoff for next run of challenger accordingly

        !Only applicable if self.run_obj_time

        !runs on incumbent should be superset of the runs performed for the
         challenger

        Parameters
        ----------
        challenger : Configuration
            Configuration which challenges incumbent
        incumbent : Configuration
            Best configuration so far
        run_history : RunHistory
            Stores all runs we ran so far
        inc_sum_cost: float
            Sum of runtimes of all incumbent runs

        Returns
        -------
        cutoff: int
            Adapted cutoff
        """

        if not self.run_obj_time:
            return self.cutoff

        # cost used by challenger for going over all its runs
        # should be subset of runs of incumbent (not checked for efficiency
        # reasons)
        chall_inst_seeds = run_history.get_runs_for_config(challenger)
        chal_sum_cost = sum_cost(config=challenger,
                                 instance_seed_pairs=chall_inst_seeds,
                                 run_history=run_history)
        cutoff = min(self.cutoff,
                     inc_sum_cost * self.adaptive_capping_slackfactor -
                     chal_sum_cost
                     )
        return cutoff

    def _compare_configs(self, incumbent: Configuration,
                         challenger: Configuration,
                         run_history: RunHistory,
                         aggregate_func: typing.Callable,
                         log_traj: bool = True):
        """
        Compare two configuration wrt the runhistory and return the one which
        performs better (or None if the decision is not safe)

        Decision strategy to return x as being better than y:
            1. x has at least as many runs as y
            2. x performs better than y on the intersection of runs on x and y

        Implicit assumption:
            Challenger was evaluated on the same instance-seed pairs as
            incumbent

        Parameters
        ----------
        incumbent: Configuration
            Current incumbent
        challenger: Configuration
            Challenger configuration
        run_history: RunHistory
            Stores all runs we ran so far
        aggregate_func: typing.Callable
            Aggregate performance across instances
        log_traj: bool
            Whether to log changes of incumbents in trajectory

        Returns
        -------
        None or better of the two configurations x,y
        """

        inc_runs = run_history.get_runs_for_config(incumbent)
        chall_runs = run_history.get_runs_for_config(challenger)
        to_compare_runs = set(inc_runs).intersection(chall_runs)

        # performance on challenger runs
        chal_perf = aggregate_func(challenger, run_history, to_compare_runs)
        inc_perf = aggregate_func(incumbent, run_history, to_compare_runs)

        # Line 15
        if chal_perf > inc_perf and len(chall_runs) >= self.minR:
            # Incumbent beats challenger
            self.logger.debug("Incumbent (%.4f) is better than challenger "
                              "(%.4f) on %d runs." %
                              (inc_perf, chal_perf, len(chall_runs)))
            return incumbent

        # Line 16
        if not set(inc_runs) - set(chall_runs):

            # no plateau walks
            if chal_perf >= inc_perf:
                self.logger.debug("Incumbent (%.4f) is at least as good as the "
                                  "challenger (%.4f) on %d runs." %
                                  (inc_perf, chal_perf, len(chall_runs)))
                return incumbent

            # Challenger is better than incumbent
            # and has at least the same runs as inc
            # -> change incumbent
            n_samples = len(chall_runs)
            self.logger.info("Challenger (%.4f) is better than incumbent (%.4f)"
                             " on %d runs." % (chal_perf, inc_perf, n_samples))
            # Show changes in the configuration
            params = sorted([(param, incumbent[param], challenger[param])
                             for param in challenger.keys()])
            self.logger.info("Changes in incumbent:")
            for param in params:
                if param[1] != param[2]:
                    self.logger.info("  %s : %r -> %r" % (param))
                else:
                    self.logger.debug("  %s remains unchanged: %r" %
                                      (param[0], param[1]))

            if log_traj:
                self.stats.inc_changed += 1
                # self.traj_logger.add_entry(train_perf=chal_perf,
                #                            incumbent_id=self.stats.inc_changed,
                #                            incumbent=challenger)
            return challenger

        # undecided
        return None
