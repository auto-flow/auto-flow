from typing import Optional, Dict

from sklearn.pipeline import Pipeline, FeatureUnion


def concat_pipeline(*args) -> Optional[Pipeline]:
    pipeline_list = []
    for node in args:
        if isinstance(node, Pipeline):
            pipeline_list.extend(node.steps)
    if pipeline_list:
        return Pipeline(pipeline_list)
    else:
        return None


def union_pipeline(preprocessors: Dict) -> Optional[Pipeline]:
    name = "feature_union"
    pipeline_list = []
    for key, value in preprocessors:
        if isinstance(value, Pipeline):
            pipeline_list.append((
                key,
                value
            ))
    if pipeline_list:
        return Pipeline([(
            name,
            FeatureUnion(pipeline_list)
        )])
    else:
        return None
