from copy import copy, deepcopy
from pickle import dumps, loads

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

weights = [0.25, 0.5, 0.25]
hp = CategoricalHyperparameter("B", ["1", "2", "3"], weights=weights)
sub_cs = ConfigurationSpace()
sub_cs.add_hyperparameter(hp)
cs = ConfigurationSpace()
cs.add_configuration_space("A", sub_cs)
print(deepcopy(sub_cs).get_hyperparameter("B").probabilities, weights)
print(copy(sub_cs).get_hyperparameter("B").probabilities, weights)
print(loads(dumps(sub_cs)).get_hyperparameter("B").probabilities, weights)
print(cs.get_hyperparameter("A:B").probabilities, weights)
print(deepcopy(cs).get_hyperparameter("A:B").probabilities, weights)
print(copy(cs).get_hyperparameter("A:B").probabilities, weights)
print(loads(dumps(cs)).get_hyperparameter("A:B").probabilities, weights)