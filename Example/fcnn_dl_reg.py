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
from sklearn.svm import SVR

from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

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
parser.add_argument('-m',dest='model',type=str,choices=['rf','xgb','knn' ,'svm'],default='xgb',help="??")
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


def mape(actual, forecast):
	mask = actual != 0
	return (np.fabs(actual - forecast)/actual)[mask].mean()
###
def calc_reg_r2_rmse_mse(y_true,y_pred):
	from scipy.stats import pearsonr
	x_r2_score = r2_score(y_true, y_pred)
	pearson_r2_score = pearsonr(y_true, y_pred)[0]**2
	x_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
	x_mse = mean_squared_error(y_true, y_pred)
	x_mae =	mean_absolute_error(y_true, y_pred)
	x_mape = mape(y_true, y_pred)	
	return {'r2_score':x_r2_score,'pearson_r2':pearson_r2_score,
	'rmse':x_rmse,'mse':x_mse,'mae':x_mae,'mape':x_mape}
###
def calc_reg_r2_rmse_mse(y_true, y_pred):
	from scipy.stats import pearsonr
	x_r2_score = r2_score(y_true, y_pred)
	pearson_r2_score = pearsonr(y_true, y_pred)[0]**2
	x_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
	x_mse = mean_squared_error(y_true, y_pred)
	x_mae = mean_absolute_error(y_true, y_pred)
	mask = y_true != 0
	x_mape = (np.fabs(y_true - y_pred)/y_true)[mask].mean()
	return {'r2_score': x_r2_score, 'pearson_r2': pearson_r2_score,
		'rmse': x_rmse, 'mse': x_mse, 'mae': x_mae, 'mape': x_mape}
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


class FCNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(FCNN, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, 1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return x

performance = []
num_epochs = 6000
hidden_size = 300
all_results = []
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
		time_str = str(time.ctime()).replace(':','-').replace(' ','_')	
		if not re_load:
			#if model_abbre=='FCNN':
			clf = 'FCNN'
			X_train, y_train = X, y
			X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
			y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  
			# Reshape to (batch_size, 1)
			X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
			y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1)
			X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
			y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1) 
			# Reshape to (batch_size, 1)
			# Initialize model, loss function, and optimizer
			input_size = X_train.shape[1]

			model = FCNN(input_size, hidden_size)
			criterion = nn.MSELoss()
			optimizer = optim.Adam(model.parameters(), lr=0.001)

			best_param ={}
			best_param["train_epoch"] = 0
			best_param["val_epoch"] = 0
			best_param["train_MSE"] = 9e8
			best_param["val_MSE"] = 9e8

			prefix_filename = csv_base

			for epoch in range(num_epochs):
				model.train()
				optimizer.zero_grad()
				outputs = model(X_train_tensor)
				loss = criterion(outputs, y_train_tensor)
				loss.backward()
				optimizer.step()

				# Validation
				model.eval()
				with torch.no_grad():
					val_outputs = model(X_valid_tensor)
					val_MSE = mean_squared_error(y_valid_tensor[:,0], val_outputs[:,0])
					train_outputs = model(X_train_tensor)		  
					train_MSE = mean_squared_error(y_train_tensor[:,0], train_outputs[:,0])
				#import pdb;pdb.set_trace()
				if train_MSE < best_param["train_MSE"]:
					best_param["train_epoch"] = epoch
					best_param["train_MSE"] = train_MSE
				if val_MSE < best_param["val_MSE"]:
					best_param["val_epoch"] = epoch
					best_param["val_MSE"] = val_MSE
					if val_MSE < 80:
						if not os.path.exists(f'saved_models/{seed}'):os.makedirs(f'saved_models/{seed}')			
						torch.save(model, f'saved_models/{seed}/model_{prefix_filename}_{time_str}.pt')
				#if (epoch - best_param["train_epoch"] >10) and (epoch - best_param["val_epoch"] >18):		
				#	break
				print(epoch, train_MSE, val_MSE)

				if (epoch+1) % 10 == 0:
					print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, train_MSE: {train_MSE}, val_MSE: {val_MSE}')

		time2 = time.time()
		time_for_running = time2 - time1

		if dlb_i==0:
			f=open('ml_reg_csv_app.csv','a');f.write(str(clf)+'\n'+param+'\n');f.close()

		# evaluate model
		best_model = torch.load(f'saved_models/{seed}/model_{prefix_filename}_{time_str}.pt')	 

		best_model_dict = best_model.state_dict()
		best_model_wts = copy.deepcopy(best_model_dict)

		model.load_state_dict(best_model_wts)
		#(best_model.align[0].weight == model.align[0].weight).all()
		
		test_outputs = model(X_test_tensor).detach().numpy()
		#import pdb;pdb.set_trace()
		train_r2 = pearsonr(y_train, train_outputs[:,0])[0]**2
		train_rmse = rmse(y_train, train_outputs[:,0])
		valid_r2 = pearsonr(y_valid, val_outputs[:,0])[0]**2
		valid_rmse = rmse(y_valid, val_outputs[:,0])
		test_r2 = pearsonr(y_test, test_outputs[:,0])[0]**2
		test_rmse = rmse(y_test, test_outputs[:,0])
		test_metrics = calc_reg_r2_rmse_mse(y_test, test_outputs[:,0])
		results = {'model':model_abbre,"seed":seed, 'fp_types':fp_types_txt,'valid_rmse':valid_rmse,
		'valid_r2':valid_r2,"test_rmse":test_rmse , "test_r2": test_r2,"time":time_for_running}
		results.update(test_metrics)

		final_res = {
					'model':'FCNN',
					'fp_types':fp_types_txt,
					#'i':i, #'ffn':ffn,
					 'seed':seed,
					 'train_rmse':train_rmse, 
					 'valid_rmse':valid_rmse,					  
					 'test_rmse':test_rmse, 
					 'train_r2':train_r2, 
					 'valid_r2':valid_r2,					  
					 'test_r2':test_r2, 			 
					 'best_epoch': best_param["val_epoch"],
					 #'batch_size':batch_size,
					}
		#import pdb;pdb.set_trace()
		final_res.update(calc_reg_r2_rmse_mse(y_test, test_outputs[:,0])) ## added on 2024.4.10

		all_results.append(final_res)
		i += 1
		pd.DataFrame([final_res]).to_csv(f'./fcnn_reg_rand_append.csv',mode='a')


pd.DataFrame(all_results).to_csv('./fcnn_reg_csv.csv',mode='a')
pd.DataFrame(all_results).groupby(['model','fp_types'])['test_r2','test_rmse'].apply(
	lambda x:x.mean()).to_csv('fcnn_reg_csv_grouped_csv.csv',mode='a')

