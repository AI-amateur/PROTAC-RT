# coding: utf-8

from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as calculate_auc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
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
import xgboost
import json
from sklearn.utils import shuffle


descriptor_list = ['Property', 'Constitution', 'Autocorr', 'Fragment', 'Charge',
	   'Estate', 'MOE', 'Connectivity', 'Topology', 'Kappa', 'Path',
	   'Matrix', 'InfoContent']

other_desc_list = ['moe206']
descriptor_list.extend(other_desc_list)

int_type_list = ['max_depth',	'n_estimators',	'min_child_weight',]



'Property;Constitution;Autocorr;Fragment;Charge;Estate;MOE;Connectivity;Topology;Kappa;Path;Matrix;InfoContent'

['MorganFP', 'RDkitFP', 'AtomPairFP', 'TorsionFP', 'AvalonFP',
	   'EstateFP', 'MACCSFP', 'PharmacoErGFP', 'PharmacoPFP', 'PubChemFP',
	   'MHFP6', 'MAP4']

'MorganFP;RDkitFP;AtomPairFP;TorsionFP;AvalonFP;EstateFP;MACCSFP;PharmacoErGFP;PharmacoPFP;PubChemFP;MHFP6;MAP4'

'MorganFP__RDkitFP__AtomPairFP__TorsionFP__AvalonFP__EstateFP__MACCSFP__PharmacoErGFP__PharmacoPFP__PubChemFP__MHFP6__MAP4'

fp_save_folder = '/home/phzd/AI/bidd-molmap_v1_3/FP_maps'
tmp_feature_dir = './tempignore'
if not os.path.exists(tmp_feature_dir):
	os.makedirs(tmp_feature_dir)

parser = ArgumentParser(description='note: no scalerY here')
parser.add_argument('-f',dest='csv_file',type=str,help='data_path')
parser.add_argument('-fp_pack',help="eg 'MOE;Path' ")
parser.add_argument('-seeds',nargs='*',default=[2**n for n in range(10)],help="??")
parser.add_argument('-save_pred',action='store_true',default=False,help="??")
parser.add_argument('-early_stop',action='store_true',default=False,help="might not good as shap")
parser.add_argument('-gpu_id',type=int,default=0,help="default 0")
parser.add_argument('-load_params',action='store_true',default=False,help="??")


args = parser.parse_args()
csv_file = args.csv_file;csv_base = csv_file.split('.')[0]
fp_pack = args.fp_pack
print('fp_pack = ',fp_pack)
save_pred=args.save_pred

random_seeds=[int(x) for x in args.seeds]

early_stop=args.early_stop
#random_seeds=[int(x) for x in args.seeds]
model_abbre = 'xgb'
gpu_id = args.gpu_id
load_params=args.load_params



df = pd.read_csv(csv_file)
#MASK = -1
smiles_col = df.columns[0]
values_col = df.columns[1:]
Y = df[values_col].astype('float').values  ## regression no need to use MASK
#Y = df[values_col].astype('float').fillna(MASK).values
if Y.shape[1] == 0:
	Y  = Y.reshape(-1, 1)
Y  = Y.ravel()

def random_split(df, random_state, split_size = [0.8, 0.1, 0.1]):
	base_indices = np.arange(len(df))
	base_indices = shuffle(base_indices, random_state = random_state)
	nb_test = int(len(base_indices) * split_size[2])
	nb_val = int(len(base_indices) * split_size[1])
	test_idx = base_indices[0:nb_test]
	valid_idx = base_indices[(nb_test):(nb_test+nb_val)]
	train_idx = base_indices[(nb_test+nb_val):len(base_indices)]
	print(len(train_idx), len(valid_idx), len(test_idx))
	return train_idx, valid_idx, test_idx


if not os.path.exists(tmp_feature_dir):
	os.makedirs(tmp_feature_dir)

def r2(y_true, y_pred):
	pcc, _ = pearsonr(y_true,y_pred)
	return pcc[0]**2

def rmse(y_true, y_pred):
	mse = mean_squared_error(y_true, y_pred)
	rmse = np.sqrt(mse)
	return rmse

def PRC_AUC(y_true, y_score):
	precision, recall, threshold  = precision_recall_curve(y_true, y_score) #PRC_AUC
	auc = calculate_auc(recall, precision)
	return auc

def ROC_AUC(y_true, y_score):
	auc = roc_auc_score(y_true, y_score)
	return auc

"""
space={'max_depth': hp.quniform("max_depth", low=3,high=9,q=1),
		'learning_rate': hp.quniform("learning_rate", low=3,high=9,q=1),  ##  0.025, 0.1
		'gamma': hp.uniform ('gamma', low=1,high=9,q=1),
		'reg_alpha' : hp.quniform('reg_alpha', low=40,high=180,1),
		'reg_lambda' : hp.uniform('reg_lambda', low=0,high=1,q=0.1),
		'colsample_bytree' : hp.uniform('colsample_bytree', low=0.5,high=1,q=0.1),
		'min_child_weight' : hp.quniform('min_child_weight', low=0, high=10, q=1),
		'n_estimators': 180,
		'seed': 0
	}
 __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100,
 	verbosity=1, silent=None, objective='reg:linear', booster='gbtree',
 	n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
 	subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1,
 	reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0,
 	seed=None, missing=None, importance_type='gain', **kwargs)
eta ： 默认是0.3，别名是 leanring_rate，更新过程中用到的收缩步长，在每次提升计算之后，算法会直接获得新特征的权重。 eta通过缩减特征的权重使提升计算过程更加保守；[0,1]
gamma：默认是0，别名是 min_split_loss，在节点分裂时，只有在分裂后损失函数的值下降了（达到gamma指定的阈值），才会分裂这个节点。gamma值越大，算法越保守（越不容易过拟合）；[0，∞]
max_depth：默认是6，树的最大深度，值越大，越容易过拟合；[0，∞]
min_child_weight：默认是1，决定最小叶子节点样本权重和，加权和低于这个值时，就不再分裂产生新的叶子节点。当它的值较大时，可以避免模型学习到局部的特殊样本。但如果这个值过高，会导致欠拟合。[0，∞]
max_delta_step：默认是0，这参数限制每颗树权重改变的最大步长。如果是 0 意味着没有约束。如果是正值那么这个算法会更保守，通常不需要设置。[0，∞]
subsample：默认是1，这个参数控制对于每棵树，随机采样的比例。减小这个参数的值算法会更加保守，避免过拟合。但是这个值设置的过小，它可能会导致欠拟合。 (0,1]
colsample_bytree：默认是1，用来控制每颗树随机采样的列数的占比； (0,1]
colsample_bylevel：默认是1，用来控制的每一级的每一次分裂，对列数的采样的占比； (0,1]
lambda：默认是1，别名是reg_lambda，L2 正则化项的权重系数，越大模型越保守；
alpha：默认是0，别名是reg_alpha，L1 正则化项的权重系数，越大模型越保守；
seed：随机数种子，相同的种子可以复现随机结果，用于调参！
n_estimators：弱学习器的数量
"""
hps_dc={}
hps_dc['xgb_regression'] = {
	'max_depth': 5,	  ### 4,5,6,8,
	'learning_rate': 0.05,  ##  0.025,0.05,0.075,0.1,0.12
	'n_estimators': 2000,  ## 750,1500,3000,5000,10000
	'gamma': 0.,			 ## 0.05,0.1,0.5,1,2,4   ## bigger in case overfit
	'min_child_weight': 5,  ## 1, 2,4,5,8,16  ## small will cause overfit; big will cause underfit
	'max_delta_step': 1,	## 0, 2, 4
	'subsample': 0.53,	  ## 0.5, 0.6,0.7,0.8,0.9,1
	'colsample_bytree': 0.66,  ##  0.5,0.8,0.9,1
	'colsample_bylevel': 1,   ##  0.5,0.7,1
	'reg_alpha': 0,		  # 0,1,2,3,4,5
	'reg_lambda': 1,		 ## 0, 0.2,0.5, 0.75,1,1.2
	'scale_pos_weight': 1,  ##
	'base_score': 0.5,
	'seed': 2016,	   ## not
	'early_stopping_rounds': 100
}

performance = []
#for fp_pack in fp_packs:
#fp_pack = fp_packs[0]

fp_types = fp_pack.split(';')
fp_types_txt = '_'.join(fp_types)
print('fp_types = ',fp_types)
for i,fp_type in enumerate(fp_types):
	print(csv_file, fp_type)
	X2_name = f'./tempignore/{fp_type}.csv'
	X2 = pd.read_csv(X2_name).values
	if i==0:tmp=X2
	else:tmp = np.hstack((tmp,X2))
X2=tmp

"""
print('seed = ',seed)
train_idx, valid_idx, test_idx = random_split(df,random_state=seed)
train_idx = [i for i in train_idx if i < len(df)]
valid_idx = [i for i in valid_idx if i < len(df)]
test_idx = [i for i in test_idx if i < len(df)]
print(len(train_idx), len(valid_idx), len(test_idx))
X = X2[train_idx];y = Y[train_idx]
X_valid = X2[valid_idx];y_valid = Y[valid_idx]
X_test = X2[test_idx];y_test = Y[test_idx]
"""


# Set up possible values of parameters to optimize over
res = []
save_model=f'{csv_base}_{model_abbre}'
print('model_abbre = ',model_abbre)
time_start_fitting = time.time()
xgb_seed = 123;#seed
"""
#clf = xgboost.XGBRegressor(gpu_id = 0,tree_method = 'gpu_hist',objective ='reg:squarederror')
clf = xgboost.XGBRegressor(gpu_id = 0,tree_method = 'gpu_hist',objective ='reg:squarederror',
		max_depth=5,#nthread=cpu_cores,
		learning_rate=0.05, n_estimators=10000, gamma=0,min_child_weight=5,max_delta_step=1,
		subsample=0.53,colsample_bytree=0.66, colsample_bylevel=1,reg_alpha=0,
		reg_lambda=1,scale_pos_weight=1, base_score=0.5,seed=xgb_seed)
"""

## 0. def pSearch dic which have a list of preset values for each param
pSearch_old = {
	'max_depth': [4,5,6,8,],   #5,
	'learning_rate': [0.025,0.05,0.075,0.1,0.12], ## 0.05,
	#'n_estimators':[750,1500,3000,5000,10000],  ##3000,  ##
	'gamma':	   [0.05,0.1,0.5,1,2,4 ],  ##  0.,  ## bigger in case overfit
	'min_child_weight':  [1,2,4,5,8,16],  ## 5,  ## small will cause overfit; big will cause underfit
	'max_delta_step': [0, 2, 4],  #1, ##
	'subsample':	[0.5, 0.6,0.7,0.8,0.9,1],   ## 0.53,
	'colsample_bytree': [0.5,0.8,0.9,1],  ## 0.66,
	'colsample_bylevel': [0.5,0.7,1],	  ##  1,
	'reg_alpha':	 [0,1,2,3,4,5],   #  0,
	'reg_lambda':	 [0,0.2,0.5,0.75,1,1.2],	## 1,
	}
	#'scale_pos_weight': 1,  ##
	#'base_score': 0.5,
	#'seed': 2016,	   ## not
	#'early_stopping_rounds': 100

pSearch = {
	'max_depth': [4,5,6,8,],   # fix 5
	'learning_rate': [0.01,0.025,0.05,0.075,0.1,0.12],  # [0.0002,0.005,0.01,0.02,0.04,0.05,0.07,0.08],## 0.05,
	'n_estimators':[500,1000,2000,3000,4000],  ##3000,  ##
	'gamma':	  [0.05,0.1,0.5,1,2,4 ],  ##  0.,  ## bigger in case overfit
	'min_child_weight': [1,2,4,5,8,16], ## 16 [1,2,4,5,8,16],  ## 5,  ## small will cause overfit; big will cause underfit
	'max_delta_step':  [0,1,2,3,4],  ## [0, 2, 4],  #1, ##
	'subsample':	[0.5,0.53,0.6,0.7,0.8,0.9,1],   ## 0.53,
	'colsample_bytree': [0.3,0.4,0.5,0.6,0.66,0.7,0.8],#[0.5,0.8,0.9,1],  ## 0.66,
	'reg_alpha':	[0,1,2,3,4,5],  #  0,  [0,1,2,3,4,5],
	'reg_lambda':   [0,0.2,0.5,0.75,1,1.2],	## 1,
	}

## 1. init best, old params
i=0
if load_params:
	param_w_file=sorted(glob.glob('params_*'))[-1]
	with open(param_w_file,'r') as f:
		best=json.load(f)
	best['params'].update({'gpu_id':gpu_id})
	c_params = best['params']
	old = {};old.update(best)   ## seeds old useless
	best_r2 = best['results']['test_r2']
	print(f'best_r2 = {best_r2}')
else:
	clf = xgboost.XGBRegressor(gpu_id = 0,tree_method = 'gpu_hist',objective ='reg:squarederror',
		max_depth=5,#nthread=cpu_cores,
		learning_rate=0.05, n_estimators=2000, gamma=0,min_child_weight=5,max_delta_step=1,
		subsample=0.53,colsample_bytree=0.66, colsample_bylevel=1,reg_alpha=0,
		reg_lambda=1,scale_pos_weight=1, base_score=0.5,seed=xgb_seed)
	c_params = clf.get_params()
	old = {'params':c_params,'results':{}}  ## seeds old useless
	best = {'params':{},'results':{}}
	best['params'].update(c_params)
	best['results'].update({"test_r2":0})
	best_r2 = 0
## 2. iter Item and Values
for item in pSearch:
	print(f'search item:{item} in {str(pSearch[item])}')
	f=open('search_params_append.csv','a');f.write(f"{item}: {str(pSearch[item]).replace(',','_')}\n");f.close()
	one_group_res = []
	for value in pSearch[item]:
		c_params.update({item:value})
		print(f'checking {item}:{value}')
		print('c_params = ',c_params)
		test_r2s = [];test_rmses = []
		for dlb_i,seed in enumerate(random_seeds):
			print('seed = ',seed)
			train_idx, valid_idx, test_idx = random_split(df,random_state=seed)
			train_idx = [i for i in train_idx if i < len(df)]
			valid_idx = [i for i in valid_idx if i < len(df)]
			test_idx = [i for i in test_idx if i < len(df)]
			print(len(train_idx), len(valid_idx), len(test_idx))
			X = X2[train_idx]; y = Y[train_idx]
			X_valid = X2[valid_idx];y_valid = Y[valid_idx]
			X_test = X2[test_idx]; y_test = Y[test_idx]
			# Set up possible values of parameters to optimize over
			res = []


			clf = xgboost.XGBRegressor(**c_params)
			time1 = time.time()
			if early_stop:
				evaluation = [(X, y), (X_valid, y_valid)]
				clf.fit(X, y, eval_set=evaluation, eval_metric="rmse",
				early_stopping_rounds=200,verbose=False)
			else:clf.fit(X, y)
			time2 = time.time()
			time_fitting = time2 - time1
			print(f'time_fitting is: {time_fitting}')
			train_r2 = pearsonr(y, clf.predict(X))[0]**2
			train_rmse = rmse(y, clf.predict(X))
			valid_r2 = pearsonr(y_valid, clf.predict(X_valid))[0]**2
			valid_rmse = rmse(y_valid, clf.predict(X_valid))
			test_r2 = pearsonr(y_test, clf.predict(X_test))[0]**2
			test_rmse = rmse(y_test, clf.predict(X_test))

			results = {'model':model_abbre,'fp_types':fp_types_txt,'seed':seed,'train_rmse':train_rmse,'train_r2':train_r2,
		"valid_rmse":valid_rmse, "valid_r2": valid_r2,"test_rmse":test_rmse, "test_r2": test_r2,
		"time":time_fitting,"time":time_fitting}
			pd.DataFrame([results]).to_csv('./search_params_append_raw.csv',mode='a')
			print('results = ',results)
		## use one_group_res to group agg: test_r2
			test_r2s.append(test_r2)
			test_rmses.append(test_rmse)
		test_r2_avg = np.mean(test_r2s)
		test_rmse_avg = np.mean(test_rmses)
		results_sum = {'model':model_abbre,'fp_types':fp_types_txt,"test_rmse":test_rmse_avg, "test_r2": test_r2_avg}

		#one_group_res.append([value,test_r2])
		one_group_res.append([value,test_r2_avg])
		#performance.append(results)
		pd.DataFrame([results_sum]).to_csv('./search_params_append.csv',mode='a')
		if test_r2_avg > best_r2:
			best_r2 = test_r2_avg
			now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
			best['params'].update(c_params)
			best['results'].update(results_sum)
			param_w_file = f'params_{now}'
			print(f'test_r2 : {test_r2_avg}, writing {param_w_file}')
			with open(param_w_file, 'w') as f:
				json.dump(best, f, indent=4, sort_keys=True)
			print(f'test_r2 reached to {test_r2}, and wrote {param_w_file}')
		del clf;gc.collect()
	## after each iter, update the local_best_params to c_params
	#c_params={};c_params.update(best['params']) ##

	dfr = pd.DataFrame(one_group_res, columns = [item,'test_r2'])
	gidx = dfr['test_r2'].idxmax()
	local_best_params = dfr.iloc[gidx].to_dict()
	print('local_best_params = ', local_best_params)
	f=open('search_params_append.csv','a');
	print(f"best {item} values is: {local_best_params[item]}, test_r2_avg is: {local_best_params['test_r2']}")
	f.write(f"best {item} values is: {local_best_params[item]}, test_r2_avg is: {local_best_params['test_r2']}\n")
	f.close()
	local_best_params.pop('test_r2')
	#int_type_list = ['max_depth',	'n_estimators',	'min_child_weight',]
	for k,v in local_best_params.items():
		if k in int_type_list:local_best_params[k]=int(v)
	c_params.update(local_best_params)

f=open('search_params_append.csv','a');
print(f"best is: {str(best)}")
f.write(f"current best is: {str(best)}\n")
f.write(f"param_w_file is: {param_w_file}\n")
f.close()

