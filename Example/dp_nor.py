# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import os,sys
from argparse import ArgumentParser
from joblib import load, dump

fingerprint_list = ['MorganFP', 'RDkitFP', 'AtomPairFP', 'TorsionFP', 'AvalonFP', 
'EstateFP', 'MACCSFP', 'PharmacoErGFP', 'PharmacoPFP', 'PubChemFP', 'MHFP6', 'MAP4']
descriptor_list = ['Property', 'Constitution', 'Autocorr', 'Fragment', 'Charge', 
'Estate', 'MOE', 'Connectivity', 'Topology', 'Kappa', 'Path', 'Matrix', 'InfoContent']

other_desc_list = ['moe206','rdkit2d']

descriptor_list.extend(other_desc_list)

tmp_feature_dir = './tempignore'
if not os.path.exists(tmp_feature_dir):
	os.makedirs(tmp_feature_dir)

parser = ArgumentParser(description='???')
parser.add_argument('-f',dest='train_file',type=str,help='csv file')
parser.add_argument('-test_files',nargs='*',help='csv files')
parser.add_argument('-fps',nargs='*',default=[],help='eg MOE Path')
parser.add_argument('-save_csv',action='store_true',default=False,help='if save csv meta file')
parser.add_argument('-overwrite',action='store_true',default=False,help='?')

args = parser.parse_args()
train_file = args.train_file;train_base = train_file.split('.')[0]
test_files = args.test_files
fps = args.fps
print('fps = ',fps)
save_csv=args.save_csv
overwrite=args.overwrite

def process_train_desc_file(csv_base,dpCols_scaler_name=''):
	fp_file=f'{tmp_feature_dir}/{csv_base}_{fp}.csv';#print(fp_file)
	dump_name = f'{tmp_feature_dir}/{csv_base}_{fp}S.job'
	if not overwrite:
		if os.path.exists(dump_name):
			dpCols_scaler_name = f'{tmp_feature_dir}/{fp}_dpCols_scalerX.job'
			if os.path.exists(dpCols_scaler_name):
				print(f'the file:{dump_name} already exists,will skip')
				return dpCols_scaler_name				
	df=pd.read_csv(fp_file)
	print('len(df),len(df.columns) = ',len(df),len(df.columns))
	## 1. del those cols only has one single value
	nunique = df.nunique()
	cols_to_drop = nunique[nunique == 1].index
	print(f'cols_to_drop = {cols_to_drop}')
	#pd.DataFrame(cols_to_drop.to_list()).to_csv('desc_cols_to_drop.csv')
	df.drop(cols_to_drop, axis=1,inplace=True)
	if save_csv: df.to_csv(f'{tmp_feature_dir}/{csv_base}_{fp}1.csv',index=False)
	## here not del those duplicate cols, not drop del inf, but use 0 replace inf
	## 4. standard 
	#scaler=MinMaxScaler()	
	df = df.replace([np.inf, -np.inf],0)
	if not dpCols_scaler_name:
		dpCols_scaler_name = f'{tmp_feature_dir}/{fp}_dpCols_scalerX.job'
	if os.path.exists(dpCols_scaler_name):
		scaler,cols_to_drop = load(dpCols_scaler_name)
	else: 
		scaler=StandardScaler()
		scaler.fit(df.values)
		dump([scaler,cols_to_drop],dpCols_scaler_name,compress=3)
	trans_val=scaler.transform(df.values)
	trans_df=pd.DataFrame(trans_val,columns=df.columns)
	if save_csv:
		trans_df.to_csv(f'{tmp_feature_dir}/{csv_base}_{fp}S.csv',index=False)
	dump(trans_df,dump_name,compress=3)
	print(f'dump_name={dump_name},dpCols_scaler_name={dpCols_scaler_name}')
	#return trans_df,scaler,dump_name,dpCols_scaler_name
	print(f'done with {csv_base}, get {dump_name}')
	return dpCols_scaler_name

def process_test_desc_file(csv_base,dpCols_scaler_name):
	fp_file=f'{tmp_feature_dir}/{csv_base}_{fp}.csv';#print(fp_file)
	dump_name = f'{tmp_feature_dir}/{csv_base}_{fp}S.job'
	scaler,cols_to_drop = load(dpCols_scaler_name)
	if not overwrite:
		if os.path.exists(dump_name):
			print(f'the file:{dump_name} already exists,will skip')
			return 
	df=pd.read_csv(fp_file)
	print('len(df),len(df.columns) = ',len(df),len(df.columns))

	df.drop(cols_to_drop, axis=1,inplace=True)
	if save_csv:
		df.to_csv(f'{tmp_feature_dir}/{csv_base}_{fp}1.csv',index=False)
	
	df = df.replace([np.inf, -np.inf],0)
	trans_val=scaler.transform(df.values)
	trans_df=pd.DataFrame(trans_val,columns=df.columns)
	if save_csv:
		trans_df.to_csv(f'{tmp_feature_dir}/{csv_base}_{fp}S.csv',index=False)
	dump(trans_df,dump_name,compress=3)
	print(f'dump_name={dump_name},dpCols_scaler_name={dpCols_scaler_name}')
	#return trans_df,scaler,dump_name,dpCols_scaler_name
	print(f'done with {csv_base}, get {dump_name}')
	return 

for fp in fps:
	if fp not in descriptor_list:
		print('only desc fp need to do this step, which fp:{fp} not belongs to;will skip this fp');continue
	csv_base = train_base
	dpCols_scaler_name = process_train_desc_file(csv_base,'')
	if str(test_files)=='None':print('no test_files, will skip');continue
	for test_file in test_files:
		test_base = test_file.split('.')[0]
		process_test_desc_file(test_base,dpCols_scaler_name)

