import logging
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as calculate_auc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import load, dump
import numpy as np
import pandas as pd
import os,sys,glob
from argparse import ArgumentParser
from scipy.stats.stats import pearsonr
import time
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import gc  ## free memory
from xgboost import XGBRegressor
import json
from sklearn.utils import shuffle

# Argument parsing
parser = ArgumentParser(description='Model training and hyperparameter optimization')
parser.add_argument('-f', dest='csv_file', type=str, help='data_path')
parser.add_argument('-fp_pack', help="eg 'MOE;Path' ")
parser.add_argument('-seeds', nargs='*', default=[2**n for n in range(10)], help="Random seeds for cross-validation")
parser.add_argument('-save_pred', action='store_true', default=False, help="Save predictions")
parser.add_argument('-early_stop', action='store_true', default=False, help="Early stopping flag")
parser.add_argument('-gpu_id', type=int, default=0, help="Default GPU ID")
parser.add_argument('-load_params', action='store_true', default=False, help="Load pre-existing parameters")
parser.add_argument('-log_file', type=str, default="beyasian_opt.txt", help="Log file to record process")
parser.add_argument('-result_file', type=str, default="best_results.csv", help="CSV file to save best parameters and scores")

args = parser.parse_args()

csv_file = args.csv_file
fp_pack = args.fp_pack
#random_seeds = [1,8,64,128,512]
random_seeds=[int(x) for x in args.seeds]
log_file = args.log_file
result_file = args.result_file

# Set up logging
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Started hyperparameter optimization...")

int_type_list = ['max_depth',  'n_estimators', 'min_child_weight',]


# Load dataset
df = pd.read_csv(csv_file)
values_col = df.columns[1:]
Y = df[values_col].astype('float').values
Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0).ravel()  # Replace NaNs and infs

# Load features
fp_types = fp_pack.split(';')
for i, fp_type in enumerate(fp_types):
    X2_name = f'./tempignore/{fp_type}.csv'
    X2_part = pd.read_csv(X2_name).values
    X2_part = np.nan_to_num(X2_part, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaNs and infs
    if i == 0:
        X2 = X2_part
    else:
        X2 = np.hstack((X2, X2_part))

X, y = X2, Y


def r2(y_true, y_pred):
    pcc, _ = pearsonr(y_true, y_pred)
    return pcc**2


# Random split function
def random_split(df, random_state, split_size=[0.8, 0.1, 0.1]):
    base_indices = np.arange(len(df))
    base_indices = shuffle(base_indices, random_state=random_state)
    nb_test = int(len(base_indices) * split_size[2])
    nb_val = int(len(base_indices) * split_size[1])
    nb_train = len(base_indices) - nb_test - nb_val
    test_idx = base_indices[:nb_test]
    valid_idx = base_indices[nb_test: nb_test + nb_val]
    train_idx = base_indices[nb_test + nb_val:]
    return train_idx, valid_idx, test_idx

predefined_params = {
    "base_score": 0.5,
    "booster": "gbtree",
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "colsample_bytree": 0.8,
    "gamma": 0.05,
    "gpu_id": 0,
    "importance_type": "gain",
    "learning_rate": 0.01,
    "max_delta_step": 1.0,
    "max_depth": 8,
    "min_child_weight": 5,
    "n_estimators": 2000,
    "n_jobs": 1,
    "objective": "reg:squarederror",
    "random_state": 0,
    "reg_alpha": 2.0,
    "reg_lambda": 0,
    "scale_pos_weight": 1,
    "seed": 123,
    "subsample": 0.7,
    "tree_method": "gpu_hist",
    "verbosity": 1
}

def train_predefined_model():
    predefined_r2_scores = []
    predefined_rmse_scores = []
    for seed in random_seeds:
        model = XGBRegressor(
            base_score = predefined_params['base_score'],
            booster=predefined_params['booster'],
            colsample_bylevel=predefined_params['colsample_bylevel'],
            colsample_bynode=predefined_params['colsample_bynode'],
            colsample_bytree = predefined_params['colsample_bytree'],
            gamma = predefined_params['gamma'],
            gpu_id = predefined_params['gpu_id'],
            importance_type=predefined_params['importance_type'],
            learning_rate = predefined_params['learning_rate'],
            max_delta_step=predefined_params['max_delta_step'],
            max_depth = predefined_params['max_depth'],
            min_child_weight = predefined_params['min_child_weight'],
            n_estimators = predefined_params['n_estimators'],
            n_jobs=predefined_params['n_jobs'],
            objective = predefined_params['objective'],
            random_state=predefined_params['random_state'],
            reg_alpha = predefined_params['reg_alpha'],
            reg_lambda = predefined_params['reg_lambda'],
            scale_pos_weight=predefined_params['scale_pos_weight'],            
            seed=predefined_params['seed'],
            subsample = predefined_params['subsample'],           
            tree_method = predefined_params['tree_method'],
            verbosity=predefined_params['verbosity'],
        )
    """    
    # 使用预定义参数训练模型    
    train_idx, valid_idx, test_idx = random_split(df,random_state=seed)
    train_idx = [i for i in train_idx if i < len(df)]
    valid_idx = [i for i in valid_idx if i < len(df)]   
    test_idx = [i for i in test_idx if i < len(df)] 
    #print(len(train_idx), len(valid_idx), len(test_idx)) 
    X = X2[train_idx]; y = Y[train_idx]
    X_valid = X2[valid_idx];y_valid = Y[valid_idx]
    X_test = X2[test_idx]; y_test = Y[test_idx]

     # 使用预定义参数训练模型
    """
    train_idx, valid_idx, test_idx = random_split(df, random_state=seed)
    train_idx = [i for i in train_idx if i < len(df)]
    valid_idx = [i for i in valid_idx if i < len(df)]   
    test_idx = [i for i in test_idx if i < len(df)]    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 计算 R² 
    current_r2 = r2(y_test, y_pred)

    predefined_r2_scores.append(current_r2)


    logging.info(f"Seed: {seed} | predefined_R²: {current_r2}")   
    

    predefined_avg_r2 = np.mean(predefined_r2_scores)   

    logging.info(f"Predefined Model - Average R²: {predefined_avg_r2}")

    return predefined_avg_r2
# 获取预定义参数的训练结果
predefined_r2  = train_predefined_model()
logging.info(f"Predefined Parameters - R2: {predefined_r2}")   

def objective(params):
    r2_scores = []

    # Perform multiple evaluations using different random seeds
    for seed in random_seeds:
        model = XGBRegressor(
            base_score = predefined_params['base_score'],
            booster=predefined_params['booster'],
            colsample_bylevel=params['colsample_bylevel'],
            colsample_bynode=predefined_params['colsample_bynode'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            gpu_id = predefined_params['gpu_id'],
            importance_type=predefined_params['importance_type'],
            learning_rate=params['learning_rate'],
            max_delta_step=params['max_delta_step'],
            max_depth=int(params['max_depth']),
            min_child_weight = predefined_params['min_child_weight'],
            n_estimators=int(params['n_estimators']),
            n_jobs=predefined_params['n_jobs'],
            objective = predefined_params['objective'],
            random_state=predefined_params['random_state'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            scale_pos_weight=predefined_params['scale_pos_weight'],            
            seed=predefined_params['seed'],
            subsample=params['subsample'],           
            tree_method = predefined_params['tree_method'],
            verbosity=predefined_params['verbosity'],
        )
        """
        train_idx, valid_idx, test_idx = random_split(df,random_state=seed)
        train_idx = [i for i in train_idx if i < len(df)]
        valid_idx = [i for i in valid_idx if i < len(df)]   
        test_idx = [i for i in test_idx if i < len(df)] 
        #print(len(train_idx), len(valid_idx), len(test_idx)) 
        X = X2[train_idx]; y = Y[train_idx]
        X_valid = X2[valid_idx];y_valid = Y[valid_idx]
        X_test = X2[test_idx]; y_test = Y[test_idx]
        """
        # Split the dataset based on the current seed
        train_idx, valid_idx, test_idx = random_split(df, random_state=seed)
        train_idx = [i for i in train_idx if i < len(df)]
        valid_idx = [i for i in valid_idx if i < len(df)]   
        test_idx = [i for i in test_idx if i < len(df)]        
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Train the model and evaluate its performance
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate R² and RMSE for the current seed
        current_r2 = r2(y_test, y_pred)

        # Store the results
        r2_scores.append(current_r2)

        # Log the result for the current iteration
        logging.info(f"Seed: {seed} | R2: {current_r2}")

    # Calculate the average R² and RMSE from all iterations
    avg_r2 = np.mean(r2_scores)

    if avg_r2 > predefined_r2 :
        logging.info(f"Optimized Parameters: {params} | R²: {avg_r2}")
        
        # 保存优化结果
        results_df = pd.DataFrame([{
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'gamma': params['gamma'],
            'reg_alpha': params['reg_alpha'],
            'reg_lambda': params['reg_lambda'],
            'colsample_bylevel': params['colsample_bylevel'],
            'max_delta_step': params['max_delta_step'],
            'average_r2': avg_r2,
        }])
        
        # 保存到 CSV 文件
        result_file = 'beyasian_rults_opt3.csv'
        if os.path.exists(result_file):
            results_df.to_csv(result_file, mode='a', header=False, index=False)
        else:
            results_df.to_csv(result_file, index=False)


    # Log the average results
    logging.info(f"Average R2: {avg_r2}")

    return {'loss': -avg_r2, 'status': STATUS_OK,'average_r2': avg_r2,'params': params}

space_1 = {
    'n_estimators': hp.quniform('n_estimators', 400, 4000, 200),
    'max_depth': hp.quniform("max_depth", 3, 9, 1),
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.12, 0.01),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1.0, 0.1),
    'gamma': hp.quniform('gamma', 0, 4, 0.05),
    'reg_alpha': hp.quniform('reg_alpha', 0, 5.0, 1.0),
    'reg_lambda': hp.quniform('reg_lambda', 0, 1.0, 0.1),
    'colsample_bylevel': hp.quniform('colsample_bylevel', 0.5, 1.0, 0.1),
    'max_delta_step': hp.quniform("max_delta_step", 0, 4, 1),
}

space_2 = {
    'n_estimators': hp.quniform('n_estimators', 400, 800, 100),
    'max_depth': hp.quniform("max_depth", 3, 7, 1),
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.05, 0.01),
    'subsample': hp.quniform('subsample', 0.5, 0.9, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 0.9, 0.1),
    'gamma': hp.quniform('gamma', 0, 0.4, 0.05),
    'reg_alpha': hp.quniform('reg_alpha', 1, 5.0, 1.0),
    'reg_lambda': hp.quniform('reg_lambda', 0.6, 1.0, 0.1),
    'colsample_bylevel': hp.quniform('colsample_bylevel', 0.7, 1.0, 0.1),
    'max_delta_step': hp.quniform("max_delta_step", 0, 3, 1),
}


space_opt1 = {
    'n_estimators': hp.quniform('n_estimators', 1800, 2200, 100),
    'max_depth': hp.quniform("max_depth", 6, 9, 1),
    'learning_rate': hp.quniform('learning_rate', 0, 0.03, 0.01),
    'subsample': hp.quniform('subsample', 0.5, 0.9, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 0.9, 0.1),
    'gamma': hp.quniform('gamma', 0, 0.20, 0.05),
    'reg_alpha': hp.quniform('reg_alpha', 0.0, 4.0, 1.0),
    'reg_lambda': hp.quniform('reg_lambda', 0, 0.3, 0.1),
    'colsample_bylevel': hp.quniform('colsample_bylevel', 0.7, 1.0, 0.1),
    'max_delta_step': hp.quniform("max_delta_step", 1, 4, 1),
}

space_opt2 = {
    'n_estimators': hp.quniform('n_estimators', 1800, 1900, 100),
    'max_depth': hp.quniform("max_depth", 8, 9, 1),
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.02, 0.01),
    'subsample': hp.quniform('subsample', 0.8, 0.9, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.8, 0.9, 0.1),
    'gamma': hp.uniform('gamma', 0.15, 0.2),
    'reg_alpha': hp.uniform('reg_alpha', 2.0, 3.0),
    'reg_lambda': hp.uniform('reg_lambda', 0, 0.1),
    'colsample_bylevel': hp.quniform('colsample_bylevel', 0.9, 1.0, 0.1),
    'max_delta_step': hp.quniform("max_delta_step", 1, 2, 1),
}

space_opt3 = {
    'n_estimators': hp.quniform('n_estimators', 1800, 1900, 100),
    'max_depth': hp.quniform("max_depth", 8, 9, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.02),
    'subsample': hp.uniform('subsample', 0.8, 0.9),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.8, 0.9),
    'gamma': hp.uniform('gamma', 0.15, 0.2),
    'reg_alpha': hp.uniform('reg_alpha', 2.0, 3.0),
    'reg_lambda': hp.uniform('reg_lambda', 0, 0.1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.9, 1.0),
    'max_delta_step': hp.quniform("max_delta_step", 1, 2, 1),
}



trials = Trials()
best = fmin(fn=objective, space=space_opt3, algo=tpe.suggest, max_evals=200, trials=trials)

# 7. 输出最佳参数
best_result = min(trials.results, key=lambda x: x['loss'])
logging.info(f"Best parameters: {best_result['params']} with R²: {-best_result['loss']}")
print(f"Best parameters: {best_result['params']} with R²: {-best_result['loss']}")

