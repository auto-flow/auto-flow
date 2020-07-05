import logging
import traceback

from autoflow.utils.logging_ import get_logger


class BaseConfigGenerator(object):
    """
    The config generator determines how new configurations are sampled. This can take very different levels of
    complexity, from random sampling to the construction of complex empirical prediction models for promising
    configurations.
    """
    def __init__(self):
        """
        Parameters
        ----------

        directory: string
            where the results are logged
        logger: opt.utils.result_logger_v??
            the logger to store the data, defaults to v1
        overwrite: bool
            whether or not existing data will be overwritten
        logger: logging.logger
            for some debug output

        """

        self.logger=get_logger(self)

    def get_config(self, budget):
        """
        function to sample a new configuration

        This function is called inside Hyperband to query a new configuration

        Parameters
        ----------
        budget: float
            the budget for which this configuration is scheduled

        returns: (config, info_dict)
            must return a valid configuration and a (possibly empty) info dict
        """
        raise NotImplementedError('This function needs to be overwritten in %s.'%(self.__class__.__name__))

    def new_result(self, job, update_model=True):
        """
        registers finished runs

        Every time a run has finished, this function should be called
        to register it with the result logger. If overwritten, make
        sure to call this method from the base class to ensure proper
        logging.


        Parameters
        ----------
        job: instance of opt.distributed.dispatcher.Job
            contains all necessary information about the job
        update_model: boolean
            determines whether a model inside the config_generator should be updated
        """
        if not job.exception is None:
            self.logger.warning("job {} failed with exception\n{}".format(job.id, job.exception))
