from hyperflow import HyperFlowEstimator

hyperflow_pipeline=HyperFlowEstimator()
ensemble_estimator,Xy_test=hyperflow_pipeline.fit_ensemble(
    task_id="task_608d5761d4c28c4cea208a0f5e83ba22",
    hdl_id="hdl_2215affa927badf430851ce424ae4394",
    trials_fetcher_params={"k":20},
    return_Xy_test=True
)
score=ensemble_estimator.score(Xy_test[0],Xy_test[1])
print(score)
print(ensemble_estimator)