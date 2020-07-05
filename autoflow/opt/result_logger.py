#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
import os

from autoflow.resource_manager.base import ResourceManager
from .structure import Job


class DatabaseResultLogger():
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager

    def __call__(self, job: Job):
        self.resource_manager._finish_trial_update_info(job.result["info"]["trial_id"], job.timestamps)

    def new_config(self, config_id, config, config_info):
        pass


class JsonResultLogger(object):
    def __init__(self, directory, overwrite=False):
        """
        convenience logger for 'semi-live-results'

        Logger that writes job results into two files (configs.json and results.json).
        Both files contain propper json objects in each line.

        This version opens and closes the files for each result.
        This might be very slow if individual runs are fast and the
        filesystem is rather slow (e.g. a NFS).

        Parameters
        ----------

        directory: string
            the directory where the two files 'configs.json' and
            'results.json' are stored
        overwrite: bool
            In case the files already exist, this flag controls the
            behavior:

                * True:   The existing files will be overwritten. Potential risk of deleting previous results
                * False:  A FileExistsError is raised and the files are not modified.
        """

        os.makedirs(directory, exist_ok=True)

        self.config_fn = os.path.join(directory, 'configs.json')
        self.results_fn = os.path.join(directory, 'results.json')

        try:
            with open(self.config_fn, 'x') as fh:
                pass
        except FileExistsError:
            if overwrite:
                with open(self.config_fn, 'w') as fh:
                    pass
            else:
                raise FileExistsError('The file %s already exists.' % self.config_fn)
        except:
            raise

        try:
            with open(self.results_fn, 'x') as fh:
                pass
        except FileExistsError:
            if overwrite:
                with open(self.results_fn, 'w') as fh:
                    pass
            else:
                raise FileExistsError('The file %s already exists.' % self.config_fn)

        except:
            raise

        self.config_ids = set()

    def new_config(self, config_id, config, config_info):
        if not config_id in self.config_ids:
            self.config_ids.add(config_id)
            with open(self.config_fn, 'a') as fh:
                fh.write(json.dumps([config_id, config, config_info]))
                fh.write('\n')

    def __call__(self, job):
        if not job.id in self.config_ids:
            # should never happen! TODO: log warning here!
            self.config_ids.add(job.id)
            with open(self.config_fn, 'a') as fh:
                fh.write(json.dumps([job.id, job.kwargs['config'], {}]))
                fh.write('\n')
        with open(self.results_fn, 'a') as fh:
            fh.write(json.dumps([job.id, job.kwargs['budget'], job.timestamps, job.result, job.exception]))
            fh.write("\n")