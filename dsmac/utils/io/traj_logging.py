import collections
import json
import logging
import typing

from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import FloatHyperparameter, IntegerHyperparameter

from generic_fs.local import LocalFS

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"

TrajEntry = collections.namedtuple(
    'TrajEntry', ['train_perf', 'incumbent_id', 'incumbent',
                  'ta_runs', 'ta_time_used', 'wallclock_time'])


class TrajLogger(object):
    """Writes trajectory logs files and creates output directory if not exists already

    Attributes
    ----------
    stats
    logger
    output_dir
    aclib_traj_fn
    old_traj_fn
    trajectory
    """
    static_file_system = LocalFS()

    def __init__(self, output_dir, stats, file_system=LocalFS()):
        """Constructor

        Parameters
        ----------
        output_dir: str
            directory for logging (or None to disable logging)
        stats: Stats()
            Stats object
        """
        self.file_system = file_system
        TrajLogger.static_file_system = file_system
        self.stats = stats
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

        self.output_dir = output_dir
        if output_dir is None or output_dir == "":
            self.output_dir = None
            self.logger.info("No output directory for trajectory logging "
                             "specified -- trajectory will not be logged.")

        else:
            if not self.file_system.isdir(output_dir):
                try:
                    self.file_system.mkdir(output_dir)
                except OSError:
                    self.logger.debug("Could not make output directory.", exc_info=1)
                    raise OSError("Could not make output directory: "
                                  "{}.".format(output_dir))

            self.old_traj_fn = self.file_system.join(output_dir, "traj_old.csv")
            if not self.file_system.isfile(self.old_traj_fn):
                txt = (
                    '"CPU Time Used","Estimated Training Performance",'
                    '"Wallclock Time","Incumbent ID",'
                    '"Automatic Configurator (CPU) Time",'
                    '"Configuration..."\n')
                self.file_system.write_txt(self.old_traj_fn, txt)

            self.aclib_traj_fn = self.file_system.join(output_dir, "traj_aclib2.json")

        self.trajectory = []

    def add_entry(self, train_perf: float, incumbent_id: int,
                  incumbent: Configuration):
        """Adds entries to trajectory files (several formats) with using the
        same timestamps for each entry

        Parameters
        ----------
        train_perf: float
            estimated performance on training (sub)set
        incumbent_id: int
            id of incumbent
        incumbent: Configuration()
            current incumbent configuration
        """
        pass
        # ta_runs = self.stats.ta_runs
        # ta_time_used = self.stats.ta_time_used
        # wallclock_time = self.stats.get_used_wallclock_time()
        # self.trajectory.append(TrajEntry(train_perf, incumbent_id, incumbent,
        #                                  ta_runs, ta_time_used, wallclock_time))
        # if self.output_dir is not None:
        #     self._add_in_old_format(train_perf, incumbent_id, incumbent,
        #                             ta_time_used, wallclock_time)
        #     self._add_in_aclib_format(train_perf, incumbent_id, incumbent,
        #                               ta_time_used, wallclock_time)

    def _add_in_old_format(self, train_perf: float, incumbent_id: int,
                           incumbent: Configuration, ta_time_used: float,
                           wallclock_time: float):
        """Adds entries to old SMAC2-like trajectory file

        Parameters
        ----------
        train_perf: float
            Estimated performance on training (sub)set
        incumbent_id: int
            Id of incumbent
        incumbent: Configuration()
            Current incumbent configuration
        ta_time_used: float
            CPU time used by the target algorithm
        wallclock_time: float
            Wallclock time used so far
        """
        pass
        # conf = []
        # for p in incumbent:
        #     if not incumbent.get(p) is None:
        #         conf.append("%s='%s'" % (p, repr(incumbent[p])))
        #
        # txt = ("%f, %f, %f, %d, %f, %s\n" % (
        #     ta_time_used,
        #     train_perf,
        #     wallclock_time,
        #     incumbent_id,
        #     wallclock_time - ta_time_used,
        #     ", ".join(conf)
        # ))
        # self.file_system.write_txt(self.old_traj_fn, txt, append=True)

    def _add_in_aclib_format(self, train_perf: float, incumbent_id: int,
                             incumbent: Configuration, ta_time_used: float,
                             wallclock_time: float):
        """Adds entries to AClib2-like trajectory file

        Parameters
        ----------
        train_perf: float
            Estimated performance on training (sub)set
        incumbent_id: int
            Id of incumbent
        incumbent: Configuration()
            Current incumbent configuration
        ta_time_used: float
            CPU time used by the target algorithm
        wallclock_time: float
            Wallclock time used so far
        """
        pass
        conf = []
        for p in incumbent:
            if not incumbent.get(p) is None:
                conf.append("%s='%s'" % (p, repr(incumbent[p])))

        traj_entry = {"cpu_time": ta_time_used,
                      "total_cpu_time": None,  # TODO: fix this
                      "wallclock_time": wallclock_time,
                      "evaluations": self.stats.ta_runs,
                      "cost": train_perf,
                      "incumbent": conf
                      }
        try:
            traj_entry["origin"] = incumbent.origin
        except AttributeError:
            traj_entry["origin"] = "UNKNOWN"
        txt = json.dumps(traj_entry) + "\n"
        self.file_system.write_txt(self.aclib_traj_fn, txt, True)

    @staticmethod
    def read_traj_aclib_format(fn: str, cs: ConfigurationSpace):
        """Reads trajectory from file

        Parameters
        ----------
        fn: str
            Filename with saved runhistory in self._add_in_aclib_format format
        cs: ConfigurationSpace
            Configuration Space to translate dict object into Confiuration object

        Returns
        -------
        trajectory: list
            Each entry in the list is a dictionary of the form
            {
            "cpu_time": float,
            "total_cpu_time": None, # TODO
            "wallclock_time": float,
            "evaluations": int
            "cost": float,
            "incumbent": Configuration
            }
        """

        trajectory = []
        fp = TrajLogger.static_file_system.read_txt(fn).splitlines()
        for line in fp:
            entry = json.loads(line)
            entry["incumbent"] = TrajLogger._convert_dict_to_config(
                entry["incumbent"], cs=cs)
            trajectory.append(entry)

        return trajectory

    @staticmethod
    def _convert_dict_to_config(config_list: typing.List[str], cs: ConfigurationSpace):
        # CAN BE DONE IN CONFIGSPACE
        """Since we save a configurations in a dictionary str->str we have to
        try to figure out the type (int, float, str) of each parameter value

        Parameters
        ----------
        config_list: typing.List[str]
            Configuration as a list of "str='str'"
        cs: ConfigurationSpace
            Configuration Space to translate dict object into Confiuration object
        """
        config_dict = {}
        for param in config_list:
            k, v = param.split("=")
            v = v.strip("'")
            hp = cs.get_hyperparameter(k)
            if isinstance(hp, FloatHyperparameter):
                v = float(v)
            elif isinstance(hp, IntegerHyperparameter):
                v = int(v)
            config_dict[k] = v

        config = Configuration(configuration_space=cs, values=config_dict)
        config.origin = "External Trajectory"

        return config
