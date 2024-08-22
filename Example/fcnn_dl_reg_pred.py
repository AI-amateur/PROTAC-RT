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
import glob

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
parser.add_argument('-f',dest='pred_file',type=str,help='??')
parser.add_argument('-fp_packs',nargs='*',default=[],help="eg 'MOE;Path'")
parser.add_argument('-m',dest='model',type=str,default='FCNN',help="??")
parser.add_argument('-seeds',nargs='*',default=[2**n for n in range(10)],help="??")


args = parser.parse_args()
pred_file = args.pred_file;pred_base = pred_file.split('.')[0]
fp_packs = args.fp_packs
print('fp_packs = ',fp_packs)
model_abbre=args.model
random_seeds=[int(x) for x in args.seeds]

df = pd.read_csv(pred_file)
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

def calc_reg_r2_rmse_mse(y_true,y_pred):
	from scipy.stats import pearsonr
	x_r2_score = r2_score(y_true, y_pred)
	x_pearson_r2_score = pearsonr(y_true, y_pred)[0]**2
	x_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
	x_mse = mean_squared_error(y_true, y_pred)
	x_mae =	mean_absolute_error(y_true, y_pred)
	x_mape = mape(y_true, y_pred)	
	return {'x_r2_score':x_r2_score,'x_pearson_r2':x_pearson_r2_score,
	'x_rmse':x_rmse,'x_mse':x_mse,'x_mae':x_mae,'x_mape':x_mape}

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
		print(pred_file, fp_type)
		X2_name = f'./tempignore/{pred_base}_{fp_type}.csv'
		X2 = pd.read_csv(X2_name).values
		if i==0:tmp=X2
		else:tmp = np.hstack((tmp,X2))
	X2=tmp
	for seed in random_seeds:
		print('seed = ',seed)
		saved_model_file = glob.glob(f'saved_models/{seed}/*.pt')[0]

		# evaluate model
		best_model = torch.load(saved_model_file)	 

		best_model_dict = best_model.state_dict()
		best_model_wts = copy.deepcopy(best_model_dict)
		input_size = X2.shape[1]
		model = FCNN(input_size, hidden_size)
		model.load_state_dict(best_model_wts)

		f=open('fcnn_reg_rand_pred.csv','a');f.write('FCNN'+'\n');f.close()

		X_ext_tensor = torch.tensor(X2, dtype=torch.float32)
		y_ext_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)  
		ext_outputs = model(X_ext_tensor).detach().numpy()
	
		final_res = {
					'model':'FCNN',
					'fp_types':fp_types_txt,
					#'i':i, #'ffn':ffn,
					 'seed':seed,		 
					 #'batch_size':batch_size,
					}
		#import pdb;pdb.set_trace()
		final_res.update(calc_reg_r2_rmse_mse(Y, ext_outputs[:,0])) ## added on 2024.4.10

		all_results.append(final_res)
		
		pd.DataFrame([final_res]).to_csv(f'./fcnn_reg_rand_pred.csv',mode='a')
		pd.DataFrame({'Y':Y, 'Y_pred':ext_outputs[:,0]}).to_csv(f'fcnn_{pred_base}_seed{seed}_pred.csv',index=False)
		i += 1

#pd.DataFrame(all_results).to_csv('./fcnn_reg_ext.csv',mode='a')
pd.DataFrame(all_results).groupby(['model','fp_types'])['x_r2_score','x_rmse'].apply(
	lambda x:x.mean()).to_csv('fcnn_reg_csv_grouped_pred.csv',mode='a')

