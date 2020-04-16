from typing import Optional, Dict

from sklearn.pipeline import Pipeline, FeatureUnion

from autoflow.pipeline.pipeline import GenericPipeline


def concat_pipeline(*args) -> Optional[GenericPipeline]:
    pipeline_list = []
    for node in args:
        if isinstance(node, Pipeline):
            pipeline_list.extend(node.steps)
    if pipeline_list:
        return GenericPipeline(pipeline_list)
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
            FeatureUnion(pipeline_list,n_jobs=1)
        )])
    else:
        return None
