from typing import List, Tuple
from numpy import ndarray
from autoflow.manager.resource_manager import ResourceManager
from autoflow.utils.klass import StrSignatureMixin
from autoflow.utils.typing import GenericEstimator


class TrainedDataFetcher(StrSignatureMixin):
    def __init__(
            self,
            task_id: str,
            hdl_id: str,
            trial_ids: List[str],
            resource_manager: ResourceManager
    ):
        self.resource_manager = resource_manager
        self.trial_ids = trial_ids
        self.hdl_id = hdl_id
        self.task_id = task_id

    def fetch(self)->Tuple[
        List[List[GenericEstimator]],
        List[List[ndarray]],
        List[List[ndarray]]
    ]:
        estimator_list, y_true_indexes_list, y_preds_list = \
            self.resource_manager.load_estimators_in_trials(self.trial_ids)
        # self.resource_manager.close_trials_db()
        return estimator_list, y_true_indexes_list, y_preds_list