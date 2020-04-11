from collections import OrderedDict
from copy import deepcopy
from typing import Union, Tuple, List, Any, Dict

from hyperflow.constants import PHASE1, PHASE2
from hyperflow.hdl.utils import get_hdl_bank, get_default_hdl_bank
from hyperflow.utils.dict import add_prefix_in_dict_keys
from hyperflow.utils.logging import get_logger
from hyperflow.utils.math import get_int_length


class HDL_Constructor():
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
                    "liblinear_svr", "libsvm_svr", "lightgbm", "random_forest", "sgd"),
            included_highR_nan_imputers=("operate.drop", {"_name": "operate.merge", "__rely_model": "boost_model"}),
            included_cat_nan_imputers=(
                    "impute.fill_cat", {"_name": "impute.fill_abnormal", "__rely_model": "boost_model"}),
            included_num_nan_imputers=(
                    "impute.fill_num", {"_name": "impute.fill_abnormal", "__rely_model": "boost_model"}),
            included_highR_cat_encoders=("operate.drop", "encode.label", "encode.cat_boost"),
            included_lowR_cat_encoders=("encode.one_hot", "encode.label", "encode.cat_boost"),

    ):
        self.included_lowR_cat_encoders = included_lowR_cat_encoders
        self.included_highR_cat_encoders = included_highR_cat_encoders
        self.included_num_nan_imputers = included_num_nan_imputers
        self.included_cat_nan_imputers = included_cat_nan_imputers
        self.included_highR_nan_imputers = included_highR_nan_imputers
        self.included_regressors = included_regressors
        self.included_classifiers = included_classifiers
        self.logger = get_logger(__name__)
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

    def __str__(self):
        return f"hyperflow.HDL_Constructor(" \
            f"DAG_descriptions={repr(self.DAG_workflow)}, " \
            f"hdl_bank_path={repr(self.hdl_bank_path)}, " \
            f"hdl_bank={repr(self.hdl_bank)}, " \
            f"included_classifiers={repr(self.included_classifiers)}, " \
            f"included_regressors={repr(self.included_regressors)}, " \
            f"included_highR_nan_imputers={repr(self.included_highR_nan_imputers)}, " \
            f"included_cat_nan_imputers={repr(self.included_cat_nan_imputers)}, " \
            f"included_num_nan_imputers={repr(self.included_num_nan_imputers)}, " \
            f"included_highR_cat_encoders={repr(self.included_highR_cat_encoders)}, " \
            f"included_lowR_cat_encoders={repr(self.included_lowR_cat_encoders)}" \
            f")"

    __repr__ = __str__

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

    def generic_recommend(self):
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

    def run(self, data_manager, random_state, highR_cat_threshold):
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
        for key in self.DAG_workflow.keys():
            if key.split("->")[-1] == "target":
                target_key = key
        estimator_values = self.DAG_workflow.pop(target_key)
        if not isinstance(estimator_values, (list, tuple)):
            estimator_values = [estimator_values]
        preprocessing_dict = {}
        mainTask = self.ml_task.mainTask
        hdl_bank = get_default_hdl_bank()
        # 遍历DAG_describe，构造preprocessing
        n_steps = len(self.DAG_workflow)
        int_len = get_int_length(n_steps)
        for i, (key, values) in enumerate(self.DAG_workflow.items()):
            if not isinstance(values, (list, tuple)):
                values = [values]
            formed_key = f"{i:0{int_len}d}{key}(choice)"
            sub_dict = {}
            for value in values:
                packages, addition_dict, is_vanilla = self.parse_item(value)
                addition_dict.update({"random_state": self.random_state})  # fixme
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

    def get_hdl(self):
        # 获取hdl
        return self.hdl
