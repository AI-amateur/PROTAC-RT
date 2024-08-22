# coding: utf-8

import sys
sys.path.append('/home/phzd/AI/bidd-molmap_v1_3')
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as calculate_auc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.utils import shuffle 
from joblib import load, dump
import numpy as np
import pandas as pd
import os,time
from molmap import feature,loadmap
from scipy.stats.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from argparse import ArgumentParser
from sklearn.metrics import r2_score
import xgboost

bitsinfo = feature.fingerprint.Extraction().bitsinfo
#fp_types = bitsinfo.Subtypes.unique()
flist_F = bitsinfo[bitsinfo.Subtypes.isin(['MACCSFP', 'PharmacoErGFP','PubChemFP'])].IDs.tolist()
bitsinfo = feature.descriptor.Extraction().bitsinfo
flist_D = bitsinfo.IDs.tolist()
['Property', 'Constitution', 'Autocorr', 'Fragment', 'Charge',
	   'Estate', 'MOE', 'Connectivity', 'Topology', 'Kappa', 'Path',
	   'Matrix', 'InfoContent']
'Property;Constitution;Autocorr;Fragment;Charge;Estate;MOE;Connectivity;Topology;Kappa;Path;Matrix;InfoContent'

['MorganFP', 'RDkitFP', 'AtomPairFP', 'TorsionFP', 'AvalonFP',
	   'EstateFP', 'MACCSFP', 'PharmacoErGFP', 'PharmacoPFP', 'PubChemFP',
	   'MHFP6', 'MAP4']

'MorganFP;RDkitFP;AtomPairFP;TorsionFP;AvalonFP;EstateFP;MACCSFP;PharmacoErGFP;PharmacoPFP;PubChemFP;MHFP6;MAP4'

'MACCSFP;PharmacoErGFP;PubChemFP'

'MorganFP__RDkitFP__AtomPairFP__TorsionFP__AvalonFP__EstateFP__MACCSFP__PharmacoErGFP__PharmacoPFP__PubChemFP__MHFP6__MAP4'



dic_DF={'D':'X2_','F':'X1_'}
#dic_flist={'D':flist_D,'F':flist_F}

fp_save_folder = '/home/phzd/AI/bidd-molmap_v1_3/FP_maps'
tmp_feature_dir = './tempignore'
if not os.path.exists(tmp_feature_dir):
	os.makedirs(tmp_feature_dir)


parser = ArgumentParser(description='???')
parser.add_argument('-f',dest='pred_file',type=str,help='??')
parser.add_argument('-fp_packs',nargs='*',default=[],help='eg MOE;Path')
parser.add_argument('-model_files',nargs='*',default=[],help="??")


args = parser.parse_args()
pred_file = args.pred_file;pred_base = pred_file.split('.')[0]
fp_packs = args.fp_packs
print('fp_packs = ',fp_packs)
model_files=args.model_files
print('model_files = ',model_files)

df = pd.read_csv(pred_file)
#MASK = -1
smiles_col = df.columns[0]
values_col = df.columns[1:]
Y = df[values_col].astype('float').values  ## regression no need to use MASK
#Y = df[values_col].astype('float').fillna(MASK).values

if not os.path.exists(tmp_feature_dir):
	os.makedirs(tmp_feature_dir)

if Y.shape[1] == 0:
	Y  = Y.reshape(-1, 1)
Y  = Y.ravel()

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
	print('fp_types = ',fp_types)	
	print('fp_types = ',fp_types)	
	for i,fp_type in enumerate(fp_types):
		print(pred_file, fp_type)
		X2_name = f'./tempignore/{pred_base}_{fp_type}.csv'
		X2 = pd.read_csv(X2_name).values
		if i==0:tmp=X2
		else:tmp = np.hstack((tmp,X2))
	X2=tmp
	for model_file in model_files:
		clf=load(model_file)
		#if Y.shape[1] >= 2:
		Y_pred = clf.predict(X2)
		#ext_r2 = pearsonr(Y, Y_pred)[0]**2
		ext_r2 = r2_score(Y, Y_pred)
		ext_rmse = rmse(Y, Y_pred)
		results = {'pred_file':pred_file,'model_file':model_file,'fp_types':fp_types_txt,
		'ext_rmse':ext_rmse,'ext_r2':ext_r2,}
		print('results = ',results)
		performance.append(results)
		fp_packs_to_name = '_'.join(fp_packs)
		#import pdb;pdb.set_trace()
		pd.DataFrame({'Y':Y, 'Y_pred':Y_pred}).to_csv(f'{model_file}_{fp_packs_to_name}.csv',index=False)
		pd.DataFrame([results]).to_csv('ml_reg_csv_append_pred.csv',mode='a')
#pd.DataFrame(performance).to_csv('ml_reg_pred_csv.csv',mode='a')
pd.DataFrame(performance).groupby(['pred_file','fp_types'])['ext_r2'].apply(
	lambda x:x.mean()).to_csv('ml_reg_csv_grouped_pred.csv',mode='a')

