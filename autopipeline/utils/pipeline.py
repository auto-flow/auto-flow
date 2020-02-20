from sklearn.pipeline import Pipeline


def concat_pipeline(*args)->Pipeline:
    pipeline_list=[]
    for node in args:
        assert isinstance(node,Pipeline)
        pipeline_list.extend(node.steps)
    return Pipeline(pipeline_list)


