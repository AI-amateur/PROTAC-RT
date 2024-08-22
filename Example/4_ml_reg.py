# coding: utf-8


from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as calculate_auc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.utils import shuffle 
from joblib import load, dump
import numpy as np
import pandas as pd
import os,sys,json
from argparse import ArgumentParser
from scipy.stats.stats import pearsonr
import time
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import gc  ## free memory
import xgboost
import json


descriptor_list = ['Property', 'Constitution', 'Autocorr', 'Fragment', 'Charge',
	   'Estate', 'MOE', 'Connectivity', 'Topology', 'Kappa', 'Path',
	   'Matrix', 'InfoContent']

other_desc_list = ['moe206']

descriptor_list.extend(other_desc_list)

['Property', 'Constitution', 'Autocorr', 'Fragment', 'Charge',
	   'Estate', 'MOE', 'Connectivity', 'Topology', 'Kappa', 'Path',
	   'Matrix', 'InfoContent']
'Property;Constitution;Autocorr;Fragment;Charge;Estate;MOE;Connectivity;Topology;Kappa;Path;Matrix;InfoContent'

['MorganFP', 'RDkitFP', 'AtomPairFP', 'TorsionFP', 'AvalonFP',
	   'EstateFP', 'MACCSFP', 'PharmacoErGFP', 'PharmacoPFP', 'PubChemFP',
	   'MHFP6', 'MAP4']

'MorganFP;RDkitFP;AtomPairFP;TorsionFP;AvalonFP;EstateFP;MACCSFP;PharmacoErGFP;PharmacoPFP;PubChemFP;MHFP6;MAP4'

'MorganFP__RDkitFP__AtomPairFP__TorsionFP__AvalonFP__EstateFP__MACCSFP__PharmacoErGFP__PharmacoPFP__PubChemFP__MHFP6__MAP4'




tmp_feature_dir = './tempignore'
if not os.path.exists(tmp_feature_dir):
	os.makedirs(tmp_feature_dir)


parser = ArgumentParser(description='???')
parser.add_argument('-f',dest='csv_file',type=str, help='data_path')
parser.add_argument('-fp_packs',nargs='*',default=[],help="eg 'MOE;Path'")
parser.add_argument('-m',dest='model',type=str,choices=['rf','xgb','knn'],default='xgb',help="??")
parser.add_argument('-seeds',nargs='*',default=[2**n for n in range(10)],help="??")
parser.add_argument('-save_model',action='store_true',default=False,help="??")
parser.add_argument('-re_load',action='store_true',default=False,help="??")
parser.add_argument('-gpu_id',type=int,default=0,help="default 0")
parser.add_argument('-param',default='',help="json param,default empty i.e. not load param")


args = parser.parse_args()
csv_file = args.csv_file;csv_base = csv_file.split('.')[0]
fp_packs = args.fp_packs
print('fp_packs = ',fp_packs)
model_abbre=args.model
print('model_abbre = ',model_abbre)
random_seeds=[int(x) for x in args.seeds]
save_model = args.save_model
re_load=args.re_load
print('save_model, re_load = ',save_model, re_load)
gpu_id = args.gpu_id
param = args.param
print('param = ',param)


df = pd.read_csv(csv_file)
MASK = -1
smiles_col = df.columns[0]
values_col = df.columns[1:]


Y = df[values_col].astype('float').fillna(MASK).values

if not os.path.exists(tmp_feature_dir):
	os.makedirs(tmp_feature_dir)

if Y.shape[1] == 0:
	Y  = Y.reshape(-1, 1)
Y  = Y.ravel()
#import pdb;pdb.set_trace()

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


def random_split(df, random_state, split_size = [0.8, 0.1, 0.1]):
	from sklearn.utils import shuffle 
	import numpy as np
	base_indices = np.arange(len(df)) 
	base_indices = shuffle(base_indices, random_state = random_state) 
	nb_test = int(len(base_indices) * split_size[2]) 
	nb_val = int(len(base_indices) * split_size[1]) 
	test_idx = base_indices[0:nb_test] 
	valid_idx = base_indices[(nb_test):(nb_test+nb_val)] 
	train_idx = base_indices[(nb_test+nb_val):len(base_indices)] 
	print(len(train_idx), len(valid_idx), len(test_idx)) 
	return train_idx, valid_idx, test_idx 


performance = []

for fp_pack in fp_packs:
	fp_types = fp_pack.split(';')
	fp_types_txt = '_'.join(fp_types)
	for i,fp_type in enumerate(fp_types):
		print(csv_file, fp_type)
		X2_name = f'./tempignore/{fp_type}.csv'
		X2 = pd.read_csv(X2_name).values
		if i==0:tmp=X2
		else:tmp = np.hstack((tmp,X2))
	X2=tmp
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

		model_file=f"{csv_base}_{model_abbre}_mPC_CC_seed{seed}"
		time1 = time.time()
		if not re_load:	
			if model_abbre=='rf':
				n_estimators_list = [500,1000,1500,3000,4000]
				#for n_estimators in tqdm(n_estimators_list):
				n_estimators = n_estimators_list[0]
				clf = RandomForestRegressor(random_state=0,n_estimators=n_estimators,n_jobs=-1)
				#import pdb;pdb.set_trace()
				clf.fit(X, y)
				score = clf.score(X_valid, y_valid)
				res.append([int(n_estimators),score])
				dfr = pd.DataFrame(res, columns = ['n_estimators','score'])
				gidx = dfr['score'].idxmax()
				best_params = dfr.iloc[gidx].to_dict()
				best_params.pop('score')
				#import pdb;pdb.set_trace()
				print('best_params = ',best_params)

			elif model_abbre=='xgb':
				xgb_seed = 123;#seed
				if not param:
					clf = xgboost.XGBRegressor(gpu_id = 0,tree_method = 'gpu_hist',objective ='reg:squarederror',
						max_depth=5,#nthread=cpu_cores,					
						learning_rate=0.05, n_estimators=2000, gamma=0,min_child_weight=5,max_delta_step=1,  
						subsample=0.53,colsample_bytree=0.66, colsample_bylevel=1,reg_alpha=0,
						reg_lambda=1,scale_pos_weight=1, base_score=0.5,seed=xgb_seed)
				else:
					with open(param,'r') as f:x_p = json.load(f)
					x_p['params'].update({'gpu_id':gpu_id})
					clf = xgboost.XGBRegressor(**x_p['params'])
			# elif model_abbre=='svr':
			#   from sklearn import svm
			#   clf =svm.SVR(verbose=True) #,kernel='rbf',n_jobs=-1
			#   clf.fit(X, y)  ## rather poor
			elif model_abbre=='knn':
				# Set up possible values of parameters to optimize over
				n_neighbors_list = np.arange(1, 15, 2)
				weights_list =  ['uniform', 'distance']
				res = []
				for n_neighbors in tqdm(n_neighbors_list, ascii=True):
					for weights in weights_list:
						clf = KNeighborsRegressor(n_neighbors=n_neighbors, weights = weights)
						clf.fit(X, y)
						score = clf.score(X_valid, y_valid)
						res.append([n_neighbors, weights, score])
				dfr = pd.DataFrame(res, columns = ['n_neighbors', 'weights', 'score'])
				gidx = dfr['score'].idxmax()
				best_params = dfr.iloc[gidx].to_dict()
				best_params.pop('score')
				print('best_params = ', best_params)
				clf = KNeighborsRegressor(**best_params)

			#clf = RandomForestRegressor(random_state=0,**best_params)
			if model_abbre=='xgb':
				evaluation = [(X, y), (X_valid, y_valid)]	
				clf.fit(X, y, eval_set=evaluation, eval_metric="rmse",
				early_stopping_rounds=200,verbose=False)

			else:clf.fit(X, y)			
			if save_model:	dump(clf,model_file,compress=3)

		else:clf=load(save_model)
		time2 = time.time()
		time_for_running = time2 - time1

		if dlb_i==0:
			f=open('ml_reg_csv_app.csv','a');f.write(str(clf)+'\n'+param+'\n');f.close()
		valid_r2 = pearsonr(y_valid, clf.predict(X_valid))[0]**2
		#valid_r2 = r2_score(y_valid, clf.predict(X_valid))
		valid_rmse = rmse(y_valid, clf.predict(X_valid))
		test_r2 = pearsonr(y_test, clf.predict(X_test))[0]**2
		#test_r2 = r2_score(y_test, clf.predict(X_test))
		test_rmse = rmse(y_test, clf.predict(X_test))
		results = {'model':model_abbre,"seed":seed, 'fp_types':fp_types_txt,'valid_rmse':valid_rmse,
		'valid_r2':valid_r2,"test_rmse":test_rmse , "test_r2": test_r2,"time":time_for_running}
		try:results.update(best_params)
		except:print('sth wrong with best_params');pass
		print('results = ',results)
		performance.append(results)
		pd.DataFrame([results]).to_csv('./ml_reg_csv_app.csv',mode='a')
if param:
	f=open('ml_reg_csv.csv','a');f.write(param+'\n');f.close()
	f=open('ml_reg_csv_grouped_csv.csv','a');f.write(param+'\n');f.close()
pd.DataFrame(performance).to_csv('./ml_reg_csv.csv',mode='a')
pd.DataFrame(performance).groupby(['model','fp_types'])['test_r2','test_rmse'].apply(
	lambda x:x.mean()).to_csv('ml_reg_csv_grouped_csv.csv',mode='a')

