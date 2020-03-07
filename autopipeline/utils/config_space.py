from copy import deepcopy
from typing import List

from ConfigSpace.configuration_space import ConfigurationSpace, Configuration


def get_default_initial_configs(phps: ConfigurationSpace, n_configs) -> List[Configuration]:
    None_name = "None:NoneType"
    phps = deepcopy(phps)
    for config in phps.get_hyperparameters():
        name: str = config.name
        if name.startswith("FE") and name.endswith("__choice__") and (None_name in config.choices):
            config.default_value = None_name

    model_choice = phps.get_hyperparameter("MHP:__choice__")
    ans = []
    for choice in model_choice.choices:
        cur_phps = deepcopy(phps)
        cur_phps.get_hyperparameter("MHP:__choice__").default_value = choice
        default = cur_phps.get_default_configuration()
        ans.append(default)
    if len(ans) < n_configs:
        ans.extend(phps.sample_configuration(n_configs - len(ans)))
    return ans
