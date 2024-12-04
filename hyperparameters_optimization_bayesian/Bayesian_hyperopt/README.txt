1.Datasets
1).The '420intrinsic.csv' file records the 420 data used for building the model.

2).The three best descriptors and the chromatographic condition descriptors are recorded in the following files:
moe206.csv
Path.csv
Charge.csv



2.Running code((need to enter the relevant environment before running.)
1).The 'bayesian_opt.py' file records the method of optimizing the relevant hyperparameters based on the R2 metric.

2).The '4_ml_reg_.py' file records the code for building the model.

3).The command to run the code is python bayesian_opt.py -f 420intrinsic.csv -fp_pack 'moe206;Path;Charge;CC' -log bayesian_1

4).The command to run the model with the optimized hyperparameters is python 4_ml_reg_.py.py -m xgb -f 420intrinsic.csv  -fp_packs '420intrinsic_mPC_CC'   -param params_best





3.Result
1).The 'bayesian_1 and bayesian_2 ' files record the process of directly optimizing hyperparameters using the Bayesian method.

2).The 'bayesian_opt1. bayesian_opt2 and bayesian_opt3 ' files record the process of further optimizing hyperparameters using the Bayesian method based on the optimized parameters.

3).The 'bayesian_results_opt2.csv and bayesian_results_opt3.csv' files record the process of surpassing the originally optimized hyperparameters using Bayesian optimization.

2).The 'params_best' records the best hyperparameters after beyasian optimization.

3).The 'ml_reg_csv_app.csv' file records the predictive performance of the XGBoost model after applying the best hyperparameters.(the results of running the model ten times under ten random seeds.)