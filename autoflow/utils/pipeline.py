from typing import Optional, Dict

from sklearn.pipeline import Pipeline, FeatureUnion

from autoflow.workflow.ml_workflow import ML_Workflow


def concat_pipeline(*args) -> Optional[ML_Workflow]:
    pipeline_list = []
    resource_manager = None
    should_store_intermediate_result = False
    for node in args:
        if isinstance(node, ML_Workflow):
            pipeline_list.extend(node.steps)
            resource_manager = node.resource_manager
            should_store_intermediate_result = node.should_store_intermediate_result
    if pipeline_list:
        return ML_Workflow(pipeline_list, should_store_intermediate_result, resource_manager)
    else:
        return None


def purify_node(node):
    if isinstance(node, Pipeline) and len(node) == 1:
        return node[0]
    return node


def union_pipeline(preprocessors: Dict) -> Optional[Pipeline]:
    name = "feature_union"
    pipeline_list = []
    for key, value in preprocessors.items():
        if isinstance(value, Pipeline):
            pipeline_list.append((
                key,
                purify_node(value)
            ))
    if pipeline_list:
        return Pipeline([(
            name,
            FeatureUnion(pipeline_list, n_jobs=1)
        )])
    else:
        return None
