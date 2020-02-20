import logging
import re
import typing
from collections import defaultdict

from dsmac.configspace import ConfigurationSpace
from dsmac.runhistory.runhistory import RunHistory

RUNHISTORY_FILEPATTERN = 'runhistory.json'
RUNHISTORY_RE = r'runhistory\.json$'
VALIDATEDRUNHISTORY_RE = r'validated_runhistory\.json$'


class PSMAC_VALUE:
    '''global value, get last id of one dir, reduce traversal time.'''
    dir2id = defaultdict(set)


def read(run_history: RunHistory,
         output_dirs: typing.Union[str, typing.List[str]],
         configuration_space: ConfigurationSpace,
         logger: logging.Logger):
    """Update runhistory with run results from concurrent runs of pSMAC.

    Parameters
    ----------
    run_history : dsmac.runhistory.RunHistory
        RunHistory object to be updated with run information from runhistory
        objects stored in the output directory.
    output_dirs : typing.Union[str,typing.List[str]]
        List of SMAC output directories
        or Linux path expression (str) which will be casted into a list with
        file_system.glob(). This function will search the output directories
        for files matching the runhistory regular expression.
    configuration_space : ConfigSpace.ConfigurationSpace
        A ConfigurationSpace object to check if loaded configurations are valid.
    logger : logging.Logger
    """
    numruns_in_runhistory = len(run_history.data)
    initial_numruns_in_runhistory = numruns_in_runhistory
    file_system = run_history.file_system
    if isinstance(output_dirs, str):
        parsed_output_dirs = file_system.glob(output_dirs)
        if file_system.glob(run_history.file_system.join(output_dirs, "run_*")):
            parsed_output_dirs += file_system.glob(file_system.join(output_dirs, "run_*"))
    else:
        parsed_output_dirs = output_dirs

    for output_directory in parsed_output_dirs:
        for file_in_output_directory in file_system.listdir(output_directory):
            match = re.match(RUNHISTORY_RE, file_in_output_directory)
            valid_match = re.match(VALIDATEDRUNHISTORY_RE, file_in_output_directory)
            if match or valid_match:
                last_id = PSMAC_VALUE.dir2id[output_directory]
                runhistory_file = file_system.join(output_directory,
                                                   file_in_output_directory)
                updated_id_set = run_history.update_from_json(runhistory_file,
                                                              configuration_space,
                                                              id_set=PSMAC_VALUE.dir2id[output_directory],
                                                              file_system=run_history.file_system)
                PSMAC_VALUE.dir2id[output_directory] = updated_id_set
                # print(PSMAC_VALUE.dir2id)
                new_numruns_in_runhistory = len(run_history.data)
                difference = new_numruns_in_runhistory - numruns_in_runhistory
                logger.debug('Shared model mode: Loaded %d new runs from %s' % (difference, runhistory_file))
                numruns_in_runhistory = new_numruns_in_runhistory

    difference = numruns_in_runhistory - initial_numruns_in_runhistory
    logger.info('Shared model mode: Finished loading new runs, found %d new runs.' % difference)


def write(run_history: RunHistory, output_directory: str, logger: logging.Logger):
    """Write the runhistory to the output directory.

    Overwrites previously outputted runhistories.

    Parameters
    ----------
    run_history : ~dsmac.runhistory.runhistory.RunHistory
        RunHistory object to be saved.

    output_directory : str

    logger : logging.Logger
    """
    file_system = run_history.file_system
    output_filename = file_system.join(output_directory, RUNHISTORY_FILEPATTERN)

    logging.debug("Saving runhistory to %s" % output_filename)

    run_history.save_json(output_filename, save_external=False)
