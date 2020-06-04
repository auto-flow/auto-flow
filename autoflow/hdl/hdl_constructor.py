from collections import OrderedDict
from copy import deepcopy
from typing import Union, Tuple, List, Any, Dict

import numpy as np
import pandas as pd
from frozendict import frozendict

from autoflow.constants import PHASE1, PHASE2, SERIES_CONNECT_LEADER_TOKEN, SERIES_CONNECT_SEPARATOR_TOKEN
from autoflow.hdl.smac import _encode
from autoflow.hdl.utils import get_hdl_bank, get_default_hdl_bank
from autoflow.utils.dict_ import add_prefix_in_dict_keys, sort_dict
from autoflow.utils.graphviz import ColorSelector
from autoflow.utils.klass import StrSignatureMixin
from autoflow.utils.logging_ import get_logger
from autoflow.utils.math_ import get_int_length
from autoflow.utils.packages import get_class_object_in_pipeline_components


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
            hdl_metadata=frozendict(),
            included_classifiers=(
                    "adaboost", "catboost", "decision_tree", "extra_trees", "gaussian_nb", "k_nearest_neighbors",
                    "liblinear_svc", "libsvm_svc", "lightgbm", "logistic_regression", "random_forest", "sgd"),
            included_regressors=(
                    "adaboost", "bayesian_ridge", "catboost", "decision_tree", "elasticnet", "extra_trees",
                    "gaussian_process", "k_nearest_neighbors", "kernel_ridge",
                    "liblinear_svr", "lightgbm", "random_forest", "sgd"),
            included_highR_nan_imputers=("operate.drop", "operate.keep_going"),
            included_nan_imputers=(
                    "impute.adaptive_fill",),
            included_highR_cat_encoders=("operate.drop", "encode.ordinal", "encode.cat_boost"),
            included_cat_encoders=("encode.one_hot", "encode.ordinal", "encode.cat_boost"),
            num2purified_workflow=frozendict({
                "num->scaled": ["scale.standardize", "operate.keep_going"],
                "scaled->purified": ["operate.keep_going", "transform.power"]
            }),
            text2purified_workflow=frozendict({
                "text->tokenized": "text.tokenize.simple",
                "tokenized->purified": [
                    "text.topic.tsvd",
                    "text.topic.lsi",
                    "text.topic.nmf",
                ]
            }),
            date2purified_workflow=frozendict({
            }),
            purified2final_workflow=frozendict({
                "purified->final": ["operate.keep_going"]
            })

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
        self.date2purified_workflow = date2purified_workflow
        self.text2purified_workflow = text2purified_workflow
        self.purified2final_workflow = purified2final_workflow
        self.num2purified_workflow = num2purified_workflow
        self.hdl_metadata = dict(hdl_metadata)
        self.included_cat_encoders = included_cat_encoders
        self.included_highR_cat_encoders = included_highR_cat_encoders
        self.included_nan_imputers = included_nan_imputers
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
        value = deepcopy(value)
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
        packages: list = packages.split(SERIES_CONNECT_SEPARATOR_TOKEN)
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
                result.update(add_prefix_in_dict_keys(params, package + SERIES_CONNECT_LEADER_TOKEN))
            return result

    def generic_recommend(self) -> Dict[str, List[Union[str, Dict[str, Any]]]]:
        '''
        Recommend a generic DAG workflow-space.

        Returns
        -------
        DAG_workflow: dict

        '''
        DAG_workflow = OrderedDict()
        contain_highR_nan = False
        essential_feature_groups = self.data_manager.essential_feature_groups
        nan_column2essential_fg = self.data_manager.nan_column2essential_fg
        fg_set = set(essential_feature_groups)
        nan_fg_set = set(nan_column2essential_fg.values())
        # --------Start imputing missing(nan) value--------------------
        if "highR_nan" in self.data_manager.feature_groups:
            DAG_workflow["highR_nan->nan"] = self.included_highR_nan_imputers
            contain_highR_nan = True
        if contain_highR_nan or "nan" in self.data_manager.feature_groups:
            DAG_workflow["nan->imputed"] = self.included_nan_imputers
            if len(nan_fg_set) > 1:
                sorted_nan_column2essential_fg = sort_dict(nan_column2essential_fg)
                sorted_nan_fg = sort_dict(list(nan_fg_set))
                DAG_workflow[f"imputed->{','.join(sorted_nan_fg)}"] = {"_name": "operate.split",
                                                                       "column2fg": _encode(
                                                                           sorted_nan_column2essential_fg)}
            elif len(nan_fg_set) == 1:
                elem = list(nan_fg_set)[0]
                DAG_workflow[f"imputed->{elem}"] = "operate.keep_going"
            else:
                raise NotImplementedError
        # --------Start encoding categorical(cat) value --------------------
        if "cat" in fg_set:
            DAG_workflow["cat->purified"] = self.included_cat_encoders
        if "highR_cat" in fg_set:
            DAG_workflow["highR_cat->purified"] = self.included_highR_cat_encoders
        # --------processing text features--------------------
        if "text" in fg_set:
            for k, v in self.text2purified_workflow.items():
                DAG_workflow[k] = v
        # --------processing numerical features--------------------
        if "num" in fg_set:
            for k, v in self.num2purified_workflow.items():
                DAG_workflow[k] = v
        # --------finally processing--------------------
        for k, v in self.purified2final_workflow.items():
            DAG_workflow[k] = v
        # --------Start estimating--------------------
        mainTask = self.ml_task.mainTask
        if mainTask == "classification":
            DAG_workflow["final->target"] = self.included_classifiers
        elif mainTask == "regression":
            DAG_workflow["final->target"] = self.included_regressors
        else:
            raise NotImplementedError
        # todo: 如果特征多，做特征选择或者降维。如果特征少，做增维
        # todo: 处理样本不平衡
        return DAG_workflow

    def draw_workflow_space(
            self,
            colorful=True,
            candidates_colors=("#663366", "#663300", "#666633", "#333366", "#660033"),
            feature_groups_colors=("#0099CC", "#0066CC", "#339933", "#FFCC33", "#33CC99", "#FF0033", "#663399",
                                   "#FF6600")
    ):
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
        cand2c = ColorSelector(list(candidates_colors), colorful)
        feat2c = ColorSelector(list(feature_groups_colors), colorful)

        def parsed_algorithms(algorithms):
            parsed_algorithms = []
            for algorithm in algorithms:
                if isinstance(algorithm, dict):
                    parsed_algorithms.append(algorithm["_name"])
                else:
                    parsed_algorithms.append(algorithm)
            return parsed_algorithms

        def get_node_label(node_name):
            if colorful:
                return f'''<<font color="{feat2c[node_name]}">{node_name}</font>>'''
            else:
                return node_name

        def get_multi_out_edge_label(label_name, tail):
            if colorful:
                return f'''<{label_name}: <font color="{feat2c[tail]}">{tail}</font>>'''
            else:
                return f'''{label_name}: {tail}'''

        def get_single_algo_edge_label(algorithm):
            if colorful:
                return f'<<font color="{cand2c[algorithm]}">{algorithm}</font>>'
            else:
                return algorithm

        def get_algo_selection_edge_label(algorithms):
            if colorful:
                return "<{" + ", ".join(
                    [f'<font color="{cand2c[algorithm]}">{algorithm}</font>'
                     for algorithm in algorithms]) + "}>"
            else:
                return "{" + ", ".join(algorithms) + "}"

        try:
            from graphviz import Digraph
        except Exception as e:
            self.logger.warning("Cannot import graphviz!")
            self.logger.error(str(e))
            return None
        DAG_workflow = deepcopy(self.DAG_workflow)
        graph = Digraph("workflow_space")
        graph.node("data")
        for parsed_node in pd.unique(self.data_manager.feature_groups):
            graph.node(parsed_node, color=feat2c[parsed_node], label=get_node_label(parsed_node))
            graph.edge("data", parsed_node,
                       label=get_multi_out_edge_label("data_manager", parsed_node))
        for indicate, algorithms in DAG_workflow.items():
            if "->" not in indicate:
                continue
            _from, _to = indicate.split("->")
            graph.node(_from, color=feat2c[_from], label=get_node_label(_from))
            algorithms = parsed_algorithms(algorithms)
            if len(_to.split(",")) > 1:
                algorithm = algorithms[0]
                tails = []
                for item in _to.split(","):
                    tails.append(item.split("=")[-1])
                for tail in tails:
                    graph.node(tail, color=feat2c[tail], label=get_node_label(tail))
                    graph.edge(_from, tail, get_multi_out_edge_label(algorithm, tail))
            else:
                graph.node(_to, color=feat2c[_to], label=get_node_label(_to))
                if len(algorithms) == 1:
                    edge_label = get_single_algo_edge_label(algorithms[0])
                else:
                    edge_label = get_algo_selection_edge_label(algorithms)
                graph.edge(_from, _to, edge_label)
        graph.attr(label=r'WorkFlow Space')
        return graph

    def purify_step_name(self, step: str):
        # autoflow/evaluation/train_evaluator.py:252
        cnt = ""
        ix = 0
        for i, c in enumerate(step):
            if c.isdigit():
                cnt += c
            else:
                ix = i
                break
        cnt = int(cnt) if cnt else -1
        step = step[ix:]
        ignored_suffixs = ["(choice)"]
        for ignored_suffix in ignored_suffixs:
            if step.endswith(ignored_suffix):
                step = step[:len(step) - len(ignored_suffix)]
        return step

    def get_hdl_dataframe(self) -> pd.DataFrame:
        preprocessing = self.hdl.get(PHASE1)
        dicts = []
        tuple_multi_index = []

        def push(step, algorithm_selection, hyper_params):
            step = self.purify_step_name(step)
            if hyper_params:
                for hp_name, hp_dict in hyper_params.items():
                    tuple_multi_index.append((step, algorithm_selection, hp_name))
                    if isinstance(hp_dict, dict):
                        dicts.append({"_type": hp_dict.get("_type"),
                                      "_value": hp_dict.get("_value"),
                                      "_default": hp_dict.get("_default")})
                    else:
                        dicts.append({"_type": f"constant {hp_dict.__class__.__name__}",
                                      "_value": hp_dict,
                                      "_default": ""})
            else:
                tuple_multi_index.append((step, algorithm_selection, ""))
                dicts.append({"_type": "", "_value": "", "_default": ""})

        if preprocessing is not None:
            for step, algorithm_selections in preprocessing.items():
                for algorithm_selection, hyper_params in algorithm_selections.items():
                    push(step, algorithm_selection, hyper_params)
        step = PHASE2
        algorithm_selections = self.hdl[f"{step}(choice)"]
        for algorithm_selection, hyper_params in algorithm_selections.items():
            push(step, algorithm_selection, hyper_params)
        df = pd.DataFrame(dicts)
        multi_index = pd.MultiIndex.from_tuples(tuple_multi_index,
                                                names=["step", "algorithm_selections", "hyper_param_name"])
        df.index = multi_index
        return df

    def interactive_display_workflow_space(self):
        try:
            from IPython.display import display
        except Exception as e:
            self.logger.warning("Cannot import IPython")
            self.logger.error(str(e))
            return None
        display(self.draw_workflow_space())
        display(self.get_hdl_dataframe())

    def run(self, data_manager, model_registry=None):
        '''

        Parameters
        ----------
        data_manager: :class:`autoflow.manager.data_manager.DataManager`
        highR_cat_threshold: float

        '''
        if model_registry is None:
            model_registry={}
        self.data_manager = data_manager
        self.ml_task = data_manager.ml_task
        self.highR_cat_threshold = data_manager.highR_cat_threshold
        self.highR_nan_threshold = data_manager.highR_nan_threshold
        self.consider_ordinal_as_cat = data_manager.consider_ordinal_as_cat
        if isinstance(self.DAG_workflow, str):
            if self.DAG_workflow == "generic_recommend":
                self.hdl_metadata.update({"source": "generic_recommend"})
                self.logger.info("Using 'generic_recommend' method to initialize a generic DAG_workflow, \n"
                                 "to Adapt to various data such like NaN and categorical features.")
                self.DAG_workflow = self.generic_recommend()
            else:
                raise NotImplementedError
        elif isinstance(self.DAG_workflow, dict):
            self.hdl_metadata.update({"source": "user_defined"})
            self.logger.info("DAG_workflow is specifically set by user.")
        else:
            raise NotImplementedError

        target_key = None
        self.purify_DAG_describe()

        # ---- 开始将 DAG_workflow 解析成 HDL
        DAG_workflow = deepcopy(self.DAG_workflow)
        for step in DAG_workflow.keys():
            if step.split("->")[-1] == "target":
                target_key = step
        estimator_values = DAG_workflow.pop(target_key)
        if not isinstance(estimator_values, (list, tuple)):
            estimator_values = [estimator_values]
        preprocessing_dict = {}
        mainTask = self.ml_task.mainTask
        hdl_bank = deepcopy(self.hdl_bank)
        # 遍历DAG_describe，构造preprocessing
        n_steps = len(DAG_workflow)
        int_len = get_int_length(n_steps)
        for i, (step, values) in enumerate(DAG_workflow.items()):
            formed_key = f"{i:0{int_len}d}{step}(choice)"
            sub_dict = {}
            for value in values:
                packages, addition_dict, is_vanilla = self.parse_item(value)
                assert get_class_object_in_pipeline_components("preprocessing", packages, model_registry) is not None,\
                    f"In step '{step}', user defined packege : '{packages}' does not exist!"
                # todo: 适配用户自定义模型
                params = {} if is_vanilla else self.get_params_in_dict(hdl_bank, packages, PHASE1, mainTask)
                sub_dict[packages] = params
                sub_dict[packages].update(addition_dict)
            preprocessing_dict[formed_key] = sub_dict
        # 构造estimator
        estimator_dict = {}
        for estimator_value in estimator_values:
            packages, addition_dict, is_vanilla = self.parse_item(estimator_value)
            assert get_class_object_in_pipeline_components(data_manager.ml_task.mainTask, packages, model_registry) is not None, \
                f"In step '{target_key}', user defined packege : '{packages}' does not exist!"
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
