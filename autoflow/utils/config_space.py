import random
from copy import deepcopy
from typing import List, Union, Dict

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, CategoricalHyperparameter, OrdinalHyperparameter, Constant, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from autoflow.constants import PHASE1,PHASE2
from autoflow.hdl.smac import _encode


def get_random_initial_configs(shps: ConfigurationSpace, n_configs, random_state=42) -> List[Configuration]:
    None_name = "None:NoneType"
    shps = deepcopy(shps)
    shps.seed(random_state)
    for config in shps.get_hyperparameters():
        name: str = config.name
        if name.startswith(PHASE1) and name.endswith("__choice__") and (None_name in config.choices):  # fixme 重构之后 None_name是不是改变了？
            config.default_value = None_name

    model_choice = shps.get_hyperparameter(f"{PHASE2}:__choice__")
    result = []
    for choice in model_choice.choices:
        cur_phps = deepcopy(shps)
        cur_phps.get_hyperparameter(f"{PHASE2}:__choice__").default_value = choice
        default = cur_phps.get_default_configuration()
        result.append(default)
    if len(result) < n_configs:
        result.extend(shps.sample_configuration(n_configs - len(result)))
    elif len(result) > n_configs:
        result = random.sample(result, n_configs)
    return result


def replace_phps(shps: ConfigurationSpace, key, value):
    for hp in shps.get_hyperparameters():
        if hp.__class__.__name__ == "Constant" and hp.name.endswith(key):
            hp.value = _encode(value)


def get_grid_initial_configs(shps: ConfigurationSpace, n_configs=-1, random_state=42):
    grid_phps = ConfigSpaceGrid(shps)
    grid_configs = grid_phps.generate_grid()
    if n_configs > 0:
        random.seed(random_state)
        grid_configs = random.sample(grid_configs, n_configs)
    return grid_configs


def estimate_config_space_numbers(cs: ConfigurationSpace):
    result = 1
    for config in cs.get_hyperparameters():
        result *= (config.get_num_neighbors() + 1)
    return result


class ConfigSpaceGrid:

    def __init__(self, configuration_space: ConfigurationSpace, ):
        self.configuration_space = configuration_space

    def generate_grid(self,
                      num_steps_dict: Union[None, Dict[str, int]] = None,
                      ) -> List[Configuration]:
        """
        Generates a grid of Configurations for a given ConfigurationSpace. Can be used, for example, for grid search.

        Parameters
        ----------
        configuration_space: :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
            The Configuration space over which to create a grid of HyperParameter Configuration values. It knows the types for all parameter values.

        num_steps_dict: dict
            A dict containing the number of points to divide the grid side formed by Hyperparameters which are either of type UniformFloatHyperparameter or type UniformIntegerHyperparameter. The keys in the dict should be the names of the corresponding Hyperparameters and the values should be the number of points to divide the grid side formed by the corresponding Hyperparameter in to.

        Returns
        -------
        list
            List containing Configurations. It is a cartesian product of tuples of HyperParameter values. Each tuple lists the possible values taken by the corresponding HyperParameter. Within the cartesian product, in each element, the ordering of HyperParameters is the same for the OrderedDict within the ConfigurationSpace.
        """

        value_sets = []  # list of tuples: each tuple within is the grid values to be taken on by a Hyperparameter
        hp_names = []

        for hp_name in self.configuration_space._children['__HPOlib_configuration_space_root__']:
            value_sets.append(self.get_value_set(num_steps_dict, hp_name))
            hp_names.append(hp_name)

        unchecked_grid_pts = self.get_cartesian_product(value_sets, hp_names)
        checked_grid_pts = []
        condtional_grid_lens = []

        while len(unchecked_grid_pts) > 0:
            try:
                grid_point = Configuration(self.configuration_space, unchecked_grid_pts[0])
                checked_grid_pts.append(grid_point)
            except ValueError as e:
                value_sets = []
                hp_names = []
                new_active_hp_names = []

                for hp_name in unchecked_grid_pts[0]:  # For loop over currently active HP names
                    value_sets.append(tuple([unchecked_grid_pts[0][hp_name], ]))
                    hp_names.append(hp_name)
                    for new_hp_name in self.configuration_space._children[
                        hp_name]:  # Checks the HPs already active for their children also being active
                        if new_hp_name not in new_active_hp_names and new_hp_name not in unchecked_grid_pts[0]:
                            all_cond_ = True
                            for cond in self.configuration_space._parent_conditions_of[new_hp_name]:
                                if not cond.evaluate(unchecked_grid_pts[0]):
                                    all_cond_ = False
                            if all_cond_:
                                new_active_hp_names.append(new_hp_name)

                for hp_name in new_active_hp_names:
                    value_sets.append(self.get_value_set(num_steps_dict, hp_name))
                    hp_names.append(hp_name)
                if len(
                        new_active_hp_names) > 0:  # this check might not be needed, as there is always going to be a new active HP when in this except block?
                    new_conditonal_grid = self.get_cartesian_product(value_sets, hp_names)
                    condtional_grid_lens.append(len(new_conditonal_grid))
                    unchecked_grid_pts += new_conditonal_grid
            del unchecked_grid_pts[0]

        return checked_grid_pts

    def get_value_set(self, num_steps_dict, hp_name):
        param = self.configuration_space.get_hyperparameter(hp_name)
        if isinstance(param, (CategoricalHyperparameter)):
            return param.choices

        elif isinstance(param, (OrdinalHyperparameter)):
            return param.sequence

        elif isinstance(param, Constant):
            return tuple([param.value, ])

        elif isinstance(param, UniformFloatHyperparameter):
            if param.log:
                lower, upper = np.log([param.lower, param.upper])
            else:
                lower, upper = param.lower, param.upper

            if num_steps_dict is not None:
                num_steps = num_steps_dict[param.name]
                grid_points = np.linspace(lower, upper, num_steps)
            else:
                grid_points = np.arange(lower, upper, param.q)  # check for log and for rounding issues

            if param.log:
                grid_points = np.exp(grid_points)

            # Avoiding rounding off issues
            if grid_points[0] < param.lower:
                grid_points[0] = param.lower
            if grid_points[-1] > param.upper:
                grid_points[-1] = param.upper

            return tuple(grid_points)

        elif isinstance(param, UniformIntegerHyperparameter):
            if param.log:
                lower, upper = np.log([param.lower, param.upper])
            else:
                lower, upper = param.lower, param.upper

            if num_steps_dict is not None:
                num_steps = num_steps_dict[param.name]
                grid_points = np.linspace(lower, upper, num_steps)
            else:
                grid_points = np.arange(lower, upper, param.q)  # check for log and for rounding issues

            if param.log:
                grid_points = np.exp(grid_points)
            grid_points = grid_points.astype(int)

            # Avoiding rounding off issues
            if grid_points[0] < param.lower:
                grid_points[0] = param.lower
            if grid_points[-1] > param.upper:
                grid_points[-1] = param.upper

            return tuple(grid_points)

        else:
            raise TypeError("Unknown hyperparameter type %s" % type(param))

    def get_cartesian_product(self, value_sets, hp_names):
        grid = []
        import itertools
        for i, element in enumerate(itertools.product(*value_sets)):
            config_dict = {}
            for j, hp_name in enumerate(hp_names):
                config_dict[hp_name] = element[j]
            grid.append(config_dict)

        return grid

