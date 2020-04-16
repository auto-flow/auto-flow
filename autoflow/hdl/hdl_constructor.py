from collections import OrderedDict
from copy import deepcopy
from typing import Union, Tuple, List, Any, Dict

import pandas as pd

from autoflow.constants import PHASE1, PHASE2
from autoflow.hdl.utils import get_hdl_bank, get_default_hdl_bank
from autoflow.utils.dict import add_prefix_in_dict_keys
from autoflow.utils.graphviz import ColorSelector
from autoflow.utils.klass import StrSignatureMixin
from autoflow.utils.logging import get_logger
from autoflow.utils.math import get_int_length


class HDL_Constructor(StrSignatureMixin):
    '''
    ``HDL`` is abbreviation of Hyper-parameter Descriptions Language.
    It describes an abstract hyperparametric space that independent with concrete implementation.

    ``HDL_Constructor`` is a class who is responsible for translating dict-type ``DAG-workflow`` into ``H.D.L`` .

    If ``DAG-workflow`` didn't be explicit assign (str "generic_recommend" is default ),
    a generic ``DAG-workflow`` will be recommend by analyzing input data in doing :meth:`run`.

    And then, by using function :meth:`run` , ``DAG-workflow`` will be translated to ``HDL``.

    '''

    def __init__(
            self,
            DAG_workflow: Union[str, Dict[str, Any]] = "generic_recommend",
            hdl_bank_path=None,
            hdl_bank=None,
            included_classifiers=(
                    "adaboost", "catboost", "decision_tree", "extra_trees", "gaussian_nb", "k_nearest_neighbors",
                    "liblinear_svc", "libsvm_svc", "lightgbm", "logistic_regression", "random_forest", "sgd"),
            included_regressors=(
                    "adaboost", "bayesian_ridge", "catboost", "decision_tree", "elasticnet", "extra_trees",
                    "gaussian_process", "k_nearest_neighbors", "kernel_ridge",
                    "liblinear_svr", "lightgbm", "random_forest", "sgd"),
            included_highR_nan_imputers=("operate.drop", {"_name": "operate.merge", "__rely_model": "boost_model"}),
            included_cat_nan_imputers=(
                    "impute.fill_cat", {"_name": "impute.fill_abnormal", "__rely_model": "boost_model"}),
            included_num_nan_imputers=(
                    "impute.fill_num", {"_name": "impute.fill_abnormal", "__rely_model": "boost_model"}),
            included_highR_cat_encoders=("operate.drop", "encode.label", "encode.cat_boost"),
            included_lowR_cat_encoders=("encode.one_hot", "encode.label", "encode.cat_boost"),

    ):
        '''

        Parameters
        ----------
        DAG_workflow: str or dict, default="generic_recommend"

            directed acyclic graph (DAG) workflow to describe the machine-learning procedure.

            By default, this value is  "generic_recommend", means HDL_Constructor will analyze the training data
            to recommend a valid DAG workflow.

            If you want design DAG workflow by yourself, you can seed a dict .

        hdl_bank_path: str, default=None

            ``hdl_bank`` is a json file which contains  all the hyper-parameters of the algorithm.

            ``hdl_bank_path`` is this file's path. If it is None, ``autoflow/hdl/hdl_bank.json`` will be choosed.

        hdl_bank: dict, default=None

            If you pass param ``hdl_bank_path=None`` and pass  ``hdl_bank`` as a dict,
            program will not load ``hdl_bank.json``, it uses passed ``hdl_bank`` directly.

        included_classifiers: list or tuple

            active if ``DAG_workflow="generic_recommend"``, and all of the following params will active in such situation.

            It decides which **classifiers** will consider in the algorithm selection.

        included_regressors: list or tuple

            It decides which **regressors** will consider in the algorithm selection.

        included_highR_nan_imputers: list or tuple

            ``highR_nan`` is a feature_group, means ``NaN`` has a high ratio in a column.

            for example:

            >>> from numpy import NaN
            >>> column = [1, 2, NaN, NaN, NaN]    # nan ratio is 60% , more than 50% (default highR_nan_threshold)

            ``highR_nan_imputers`` algorithms will handle such columns contain high ratio missing value.

        included_cat_nan_imputers: list or tuple

            ``cat_nan`` is a feature_group, means a categorical feature column contains ``NaN`` value.

            for example:

            >>> column = ["a", "b", "c", "d", NaN]

            ``cat_nan_imputers`` algorithms will handle such columns.

        included_num_nan_imputers: list or tuple

            ``num_nan`` is a feature_group, means a numerical feature column contains ``NaN`` value.

            for example:

            >>> column = [1, 2, 3, 4, NaN]

            ``num_nan_imputers`` algorithms will handle such columns.

        included_highR_cat_encoders: list or tuple

            ``highR_cat`` is a feature_group, means a categorical feature column contains highly cardinality ratio.

            for example:

            >>> import numpy as np
            >>> column = ["a", "b", "c", "d", "a"]
            >>> rows = len(column)
            >>> np.unique(column).size / rows  # result is 0.8 , is higher than 0.5 (default highR_cat_ratio)
            0.8
            
            ``highR_cat_imputers`` algorithms will handle such columns.

        included_lowR_cat_encoders: list or tuple
        
            ``lowR_cat`` is a feature_group, means a categorical feature column contains lowly cardinality ratio.

            for example:

            >>> import numpy as np
            >>> column = ["a", "a", "a", "d", "a"]
            >>> rows = len(column)
            >>> np.unique(column).size / rows  # result is 0.4 , is lower than 0.5 (default lowR_cat_ratio)
            0.4
            
            ``lowR_cat_imputers`` algorithms will handle such columns.

        Attributes
        ----------
        random_state: int

        ml_task: :class:`autoflow.utils.ml_task.MLTask`

        data_manager: :class:`autoflow.manager.data_manager.DataManager`

        hdl: dict
            construct by :meth:`run`

        Examples
        ----------
        >>> import numpy as np
        >>> from autoflow.manager.data_manager import DataManager
        >>> from autoflow.hdl.hdl_constructor import  HDL_Constructor
        >>> hdl_constructor = HDL_Constructor(DAG_workflow={"num->target":["lightgbm"]},
        ...   hdl_bank={"classification":{"lightgbm":{"boosting_type":  {"_type": "choice", "_value":["gbdt","dart","goss"]}}}})
        >>> data_manager = DataManager(X_train=np.random.rand(3,3), y_train=np.arange(3))
        >>> hdl_constructor.run(data_manager, 42, 0.5)
        >>> hdl_constructor.hdl
        {'preprocessing': {}, 'estimating(choice)': {'lightgbm': {'boosting_type': {'_type': 'choice', '_value': ['gbdt', 'dart', 'goss']}}}}

        '''

        self.included_lowR_cat_encoders = included_lowR_cat_encoders
        self.included_highR_cat_encoders = included_highR_cat_encoders
        self.included_num_nan_imputers = included_num_nan_imputers
        self.included_cat_nan_imputers = included_cat_nan_imputers
        self.included_highR_nan_imputers = included_highR_nan_imputers
        self.included_regressors = included_regressors
        self.included_classifiers = included_classifiers
        self.logger = get_logger(self)
        self.hdl_bank_path = hdl_bank_path
        self.DAG_workflow = DAG_workflow
        if hdl_bank is None:
            if hdl_bank_path:
                hdl_bank = get_hdl_bank(hdl_bank_path)
            else:
                hdl_bank = get_default_hdl_bank()
        if hdl_bank is None:
            hdl_bank = {}
            self.logger.warning("No hdl_bank, will use DAG_descriptions only.")
        self.hdl_bank = hdl_bank
        self.random_state = 42
        self.ml_task = None
        self.data_manager = None

    def parse_item(self, value: Union[dict, str]) -> Tuple[str, dict, bool]:
        if isinstance(value, dict):
            packages = value.pop("_name")
            if "_vanilla" in value:
                is_vanilla = value.pop("_vanilla")
            else:
                is_vanilla = False
            addition_dict = value
        elif isinstance(value, str):
            packages = value
            addition_dict = {}
            is_vanilla = False
        elif value is None:
            packages = "None"
            addition_dict = {}
            is_vanilla = False
        else:
            raise TypeError
        return packages, addition_dict, is_vanilla

    def purify_DAG_describe(self):
        DAG_describe = {}
        for k, v in self.DAG_workflow.items():
            if not isinstance(v, (list, tuple)):
                v = [v]
            DAG_describe[k.replace(" ", "").replace("\n", "").replace("\t", "")] = v
        self.DAG_workflow = DAG_describe

    def _get_params_in_dict(self, dict_: dict, package: str) -> dict:
        result = deepcopy(dict_)
        for path in package.split("."):
            result = result.get(path, {})
        return result

    def get_params_in_dict(self, hdl_bank: dict, packages: str, phase: str, mainTask):
        assert phase in (PHASE1, PHASE2)
        packages: list = packages.split("|")
        params_list: List[dict] = [self._get_params_in_dict(hdl_bank[PHASE1], package) for package in
                                   packages[:-1]]
        last_phase_key = PHASE1 if phase == PHASE1 else mainTask
        params_list += [self._get_params_in_dict(hdl_bank[last_phase_key], packages[-1])]
        if len(params_list) == 0:
            raise AttributeError
        elif len(params_list) == 1:
            return params_list[0]
        else:
            result = {}
            for params, package in zip(params_list, packages):
                result.update(add_prefix_in_dict_keys(params, package + "."))
            return result

    def generic_recommend(self) -> Dict[str, List[Union[str, Dict[str, Any]]]]:
        '''
        Recommend a generic DAG workflow space back.

        Returns
        -------
        DAG_workflow: dict

        '''
        DAG_workflow = OrderedDict()
        contain_highR_nan = False
        contain_nan = False
        # todo: 对于num特征进行scale transform
        # --------Start imputing missing(nan) value--------------------
        if "highR_nan" in self.data_manager.feature_groups:
            DAG_workflow["highR_nan->nan"] = self.included_highR_nan_imputers
            contain_highR_nan = True
        if contain_highR_nan or "nan" in self.data_manager.feature_groups:
            # split nan to cat_nan and num_nan
            DAG_workflow["nan->{cat_name=cat_nan,num_name=num_nan}"] = "operate.split.cat_num"
            DAG_workflow["num_nan->num"] = self.included_num_nan_imputers
            DAG_workflow["cat_nan->cat"] = self.included_cat_nan_imputers
            contain_nan = True
        # --------Start encoding categorical(cat) value --------------------
        if contain_nan or "cat" in self.data_manager.feature_groups:
            DAG_workflow["cat->{highR=highR_cat,lowR=lowR_cat}"] = {"_name": "operate.split.cat",
                                                                    "threshold": self.highR_cat_threshold}
            DAG_workflow["highR_cat->num"] = self.included_highR_cat_encoders
            DAG_workflow["lowR_cat->num"] = self.included_lowR_cat_encoders
        # --------Start estimating--------------------
        mainTask = self.ml_task.mainTask
        if mainTask == "classification":
            DAG_workflow["num->target"] = self.included_classifiers
        elif mainTask == "regression":
            DAG_workflow["num->target"] = self.included_regressors
        else:
            raise NotImplementedError
        # todo: 如果特征多，做特征选择或者降维。如果特征少，做增维
        return DAG_workflow

    def draw_workflow_space(self):
        '''

        Notes
        ------
        You must install graphviz in your compute.

        if you are using Ubuntu or another debian Linux, you should run::

            $ sudo apt-get install graphviz

        You can also install graphviz by conda::

            $ conda install -c conda-forge graphviz

        Returns
        -------
        graph: :class:`graphviz.dot.Digraph`
            You can find usage of :class:`graphviz.dot.Digraph` in https://graphviz.readthedocs.io/en/stable/manual.html

        '''
        candidates_colors = ["#663366", "#663300", "#666633", "#333366", "#660033"]
        feature_groups_colors = ["#0099CC", "#0066CC", "#339933", "#FFCC33", "#33CC99", "#FF0033", "#663399", "#FF6600"]
        cand2c = ColorSelector(candidates_colors)
        feat2c = ColorSelector(feature_groups_colors)

        def parsed_candidates(candidates):
            parsed_candidates = []
            for candidate in candidates:
                if isinstance(candidate, dict):
                    parsed_candidates.append(candidate["_name"])
                else:
                    parsed_candidates.append(candidate)
            return parsed_candidates

        def get_node_label(node_name):
            return f'''<<font color="{feat2c[node_name]}">{node_name}</font>>'''

        from graphviz import Digraph
        DAG_workflow = deepcopy(self.DAG_workflow)
        graph = Digraph("workflow_space")
        graph.node("data")
        for parsed_node in pd.unique(self.data_manager.feature_groups):
            graph.node(parsed_node, color=feat2c[parsed_node], label=get_node_label(parsed_node))
            graph.edge("data", parsed_node,
                       label=f'''<data_manager: <font color="{feat2c[parsed_node]}">{parsed_node}</font>>''')
        for indicate, candidates in DAG_workflow.items():
            if "->" not in indicate:
                continue
            _from, _to = indicate.split("->")
            graph.node(_from, color=feat2c[_from], label=get_node_label(_from))
            candidates = parsed_candidates(candidates)
            if _to.startswith("{") and _to.endswith("}"):
                candidate = candidates[0]
                _to = _to[1:-1]
                tails = []
                for item in _to.split(","):
                    tails.append(item.split("=")[-1])
                for tail in tails:
                    graph.node(tail, color=feat2c[tail], label=get_node_label(tail))
                    graph.edge(_from, tail, f'''<{candidate}: <font color="{feat2c[tail]}">{tail}</font>>''')
            else:
                graph.node(_to, color=feat2c[_to], label=get_node_label(_to))
                if len(candidates) == 1:
                    candidates_str = f'<<font color="{cand2c[candidates[0]]}">{candidates[0]}</font>>'
                else:
                    candidates_str = "<{" + ", ".join(
                        [f'<font color="{cand2c[candidate]}">{candidate}</font>' for candidate in candidates]) + "}>"
                graph.edge(_from, _to, candidates_str)
        graph.attr(label=r'WorkFlow Space')
        return graph

    def run(self, data_manager, random_state, highR_cat_threshold):
        '''

        Parameters
        ----------
        data_manager: :class:`autoflow.manager.data_manager.DataManager`
        random_state: int
        highR_cat_threshold: float

        '''
        self.highR_cat_threshold = highR_cat_threshold
        self.data_manager = data_manager
        self.ml_task = data_manager.ml_task
        self.random_state = random_state
        if isinstance(self.DAG_workflow, str):
            if self.DAG_workflow == "generic_recommend":
                self.logger.info("Using 'generic_recommend' method to initialize a generic DAG_workflow, \n"
                                 "to Adapt to various data such like NaN and categorical features.")
                self.DAG_workflow = self.generic_recommend()
            else:
                raise NotImplementedError
        elif isinstance(self.DAG_workflow, dict):
            self.logger.info("DAG_workflow is specifically set by user.")
        else:
            raise NotImplementedError

        target_key = None
        self.purify_DAG_describe()

        # ---- 开始将 DAG_workflow 解析成 HDL
        DAG_workflow = deepcopy(self.DAG_workflow)
        for key in DAG_workflow.keys():
            if key.split("->")[-1] == "target":
                target_key = key
        estimator_values = DAG_workflow.pop(target_key)
        if not isinstance(estimator_values, (list, tuple)):
            estimator_values = [estimator_values]
        preprocessing_dict = {}
        mainTask = self.ml_task.mainTask
        hdl_bank = deepcopy(self.hdl_bank)
        # 遍历DAG_describe，构造preprocessing
        n_steps = len(DAG_workflow)
        int_len = get_int_length(n_steps)
        for i, (key, values) in enumerate(DAG_workflow.items()):
            formed_key = f"{i:0{int_len}d}{key}(choice)"
            sub_dict = {}
            for value in values:
                packages, addition_dict, is_vanilla = self.parse_item(value)
                params = {} if is_vanilla else self.get_params_in_dict(hdl_bank, packages, PHASE1, mainTask)
                sub_dict[packages] = params
                sub_dict[packages].update(addition_dict)
            preprocessing_dict[formed_key] = sub_dict
        # 构造estimator
        estimator_dict = {}
        for estimator_value in estimator_values:
            packages, addition_dict, is_vanilla = self.parse_item(estimator_value)
            params = {} if is_vanilla else self.get_params_in_dict(hdl_bank, packages, PHASE2, mainTask)
            estimator_dict[packages] = params
            estimator_dict[packages].update(addition_dict)
        final_dict = {
            PHASE1: preprocessing_dict,
            f"{PHASE2}(choice)": estimator_dict
        }
        self.hdl = final_dict

    def get_hdl(self) -> Dict[str, Any]:
        '''

        Returns
        -------
        hdl: dict


        '''
        # 获取hdl
        return self.hdl
