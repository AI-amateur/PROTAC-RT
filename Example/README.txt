*To run the corresponding code, you first need to enter the corresponding environment (e.g., source activate molmap). Here, 'molmap' is used as an example of the environment.


1.Datasets
1). training set:420intrinsic.csv.(random seed split to divide 420intrinsic data into train, validation, and test 8:1:1,the training set is used to build the model, the validation set is applied to adjust hyperparameters, and the test set for final evaluation of the model prediction ability. And the robustness of each model was assessed by ten different random seeds (1, 2, 4, 8, 16, 32, 64, 128, 256 and 512).)

2). external validation set: 113external.csv


2.Feature Calculation
1).The 'fp_dp_cal.py' file records the code for calculating molecular fingerprints and descriptors. The command to run the code is as follows:
python fp_dp_cal.py -f 420intrinsic.csv -fps  Charge Path 

2).After the calculation of descriptors is completed, they need to be normalized. The running code and commands are as follows:
code:dp_nor.py
run command:python dp_nor.py -f 420intrinsic.csv -fps Path Charge
*The calculated fingerprints and descriptors are all saved in the "tempignore" folder.

3).The MOE206 descriptors, after being calculated using the relevant software, are also saved in the "tempignore" folder, and the chromatographic condition descriptors (CC), after being organized, are similarly stored in the "tempignore" folder.

3.Training model
1).The integration of MOE206, Path, Charge, and CC into a combined descriptor file is also saved in the "tempignore" folder, with the file name "420intrinsic_mPC_CC.csv".

2).After the descriptors are prepared, proceed with the model construction. The running code and commands for building the model are as follows:
code:4_ml_reg.py
run command:python 4_ml_reg.py  -m xgb -f 420intrinsic.csv -fp_packs '420intrinsic_mPC_CC' -save_model
*After the execution is completed, the files "ml_reg_csv_app.csv" and "ml_reg_csv.csv" record the specific parameters and predictive performance data of the model run, while "ml_reg_csv_grouped_csv.csv" records the average predictive performance of the model. Ten files named "420intrinsic_xgb_mPC_CC_seedn" record the constructed models.

*The fcnn requires a different code, with the execution code and command being fcnn_dl_reg.py. The command to run it is python fcnn_dl_reg.py -f 420intrinsic.csv -fp_packs '420intrinsic_mPC_CC'. After the run is completed, three files named 'fcnn_reg_csv.csv', 'fcnn_reg_csv_grouped_csv.csv', 'fcnn_reg_rand_append.csv', and a folder named 'saved_models' for saving the models will be produced.


4.Model prediction
1).The same method is used to calculate and organize the descriptors of the 113external dataset, which are saved in the 'tempignore'folder.

2).Use the 10 files named "420intrinsic_xgb_mPC_CC_seedn" to predict the 113external dataset respectively (Take seed1 as an example here). The code and command for running this are as follows:
code:4_ml_reg_pred.py     fcnn_dl_reg_pred.py
run command:python 4_ml_reg_pred.py -f 113external.csv -fp_packs mPC_CC -model_files 420intrinsic_xgb_mPC_CC_seed1
            python fcnn_dl_reg_pred.py -f 113external.csv -fp_packs 'mPC_CC' -seeds 1

*After the prediction is completed, three files are generated: 'ml_reg_csv_append_pred.csv' records the specific performance of the predictions (for fcnn, it is recorded in 'fcnn_reg_rand_pred.csv'), 'ml_reg_csv_grouped_pred.csv' records the average performance of the predictions (for fcnn, it is recorded in 'fcnn_reg_csv_grouped_pred.csv'), and '420intrinsic_xgb_mPC_CC_seed1_mPC_CC.csv' records the specific RT values predicted by the model (for fcnn, it is recorded in 'fcnn_113external_seed1_pred.csv').   


