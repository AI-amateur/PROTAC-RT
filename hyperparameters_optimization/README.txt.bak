1.Datasets
1).The '420intrinsic.csv' file records the 420 data used for building the model.

2).The three best descriptors and the chromatographic condition descriptors are recorded in the following files:
moe206.csv
Path.csv
Charge.csv



2.Running code((need to enter the relevant environment before running.)
1).The 'xgb_search_param_csvRand_r2_.py' file records the method of optimizing the relevant hyperparameters based on the R2 metric.

2).The '4_ml_reg_.py' file records the code for building the model.

3).The command to run the code is python xgb_search_param_csvRand_r2_.py -f 420intrinsic.csv -fp_pack 'moe206;Path;Charge;CC'

4).The command to run the model with the optimized hyperparameters is python 4_ml_reg_.py.py -m xgb -f 420intrinsic.csv  -fp_packs '420intrinsic_mPC_CC'   -param params_2024-07-19-07_50_41





3.Result
1).The 'search_params_append.csv' file records the process of hyperparameter optimization.

2).The 'params_2024-07-19-07_50_41' records the best hyperparameters after optimization.

3).The 'ml_reg_csv_app.csv' file records the predictive performance of the XGBoost model after applying the best hyperparameters.(the results of running the model ten times under ten random seeds.)