import os,sys,time,glob

sys.path.append('/home/phzd/AI/bidd-molmap_v1_3')
from molmap.feature import descriptor
from molmap.feature import fingerprint
from molmap import feature
import pandas as pd
import numpy as np
import argparse
import csv
##==========
fingerprint_list = ['MorganFP', 'RDkitFP', 'AtomPairFP', 'TorsionFP', 'AvalonFP', 
'EstateFP', 'MACCSFP', 'PharmacoErGFP', 'PharmacoPFP', 'PubChemFP', 'MHFP6', 'MAP4']
descriptor_list = ['Property', 'Constitution', 'Autocorr', 'Fragment', 'Charge', 
'Estate',  'Connectivity', 'Topology', 'Kappa', 'Path', 'InfoContent']
bool_dtype_list = ['MorganFP','RDkitFP','AtomPairFP','TorsionFP','AvalonFP','EstateFP',
'MACCSFP','PharmacoErGFP','PharmacoPFP','PubChemFP','MHFP6','MAP4']
int_dtype_list = ['Constitution','Fragment','Estate','Path',]
float_dtype_list =['Property','Autocorr','Charge','MOE','Connectivity','Topology',
'Kappa','Matrix','InfoContent']

astype_dic = {'int':'int','bool':'int','float':np.float16}
#out_format = 'csv' ## or 'job'
#threads = 20
#overwrite = False
fp_feat_dir = './tempignore'
if not os.path.exists(fp_feat_dir):os.mkdir(fp_feat_dir)

base=os.getcwd()

def gen_one_type_fp(csv_file,s_col,fp,args):
	csv_base = csv_file.split('.')[0]
	save_name = f'{fp_feat_dir}/{csv_base}_{fp}.{args.out_format}'  
	# args.out_format:  job or csv
	print('args.overwrite = ',args.overwrite)
	if not args.overwrite:
		if os.path.exists(save_name):
			print(f'{save_name} already exists,will skip');return save_name
	df=pd.read_csv(csv_file)
	smiles_col = df.columns[int(s_col)] if s_col.isdigit() else s_col
	smiles_list= df[smiles_col]
	x_dict = {fp:{}}
	if fp in fingerprint_list:
		extractor=fingerprint.Extraction(feature_dict=x_dict)
		bitsinfo = fingerprint.Extraction().bitsinfo
		##['PubChemFP0','PubChemFP1',…]				
	elif fp in descriptor_list:
		extractor=descriptor.Extraction(feature_dict=x_dict)
		bitsinfo = descriptor.Extraction().bitsinfo	
	else: raise IOError('not supp fp: {}'.format(fp))
	if args.threads>1:
		res=extractor.batch_transform_by_multprocess(smiles_list=smiles_list, n_jobs = args.threads)
	else:
		res=extractor.batch_transform(smiles_list=smiles_list, n_jobs = args.threads)
	#n_col = len(res[0])
	#scol_names = [fp+'{:0>4}'.format(str(i)) for i in range(n_col)]
	#print('col_names =',col_names[:10])
	flist = bitsinfo[bitsinfo.Subtypes.isin([fp])].IDs.tolist()	
	"""
	if save_type=='csv_gzip':
		feat_df = pd.DataFrame(res,columns=col_names,dtype='int')
		save_name = '{}.csv.gz'.format(fp)
		feat_df.to_csv(save_name,compression='gzip',index=False)
	else:
		save_name = '{}.csv'.format(fp)
		np.savetxt(save_name,res,fmt='%d',delimiter=',')"""	
	#w_col = str(col_names)#[','.join(col_names)]
	if fp in bool_dtype_list:dtype = 'int'
	elif fp in int_dtype_list:dtype = 'int'
	elif fp in float_dtype_list:dtype = 'float'
	else: raise IOError('not supp fp')

	x_astype = astype_dic[dtype]
	print('before res =',res)
	res=np.array(res).astype(x_astype)	#('int')
	if args.out_format=='job':
		save_name = f'{fp_feat_dir}/{csv_base}_{fp}.job'
		#xfp = np.vstack((flist,res))
		from joblib import dump
		dump(res, save_name,compress=3)
		print('job type file, here no columns name')  
		## use compress=3 is good, no compress will give much bigger file
	#pd.DataFrame(res,columns=flist).to_csv(save_name,index=False) # this is good 24 MB
	elif args.out_format=='csv':
		save_name = f'{fp_feat_dir}/{csv_base}_{fp}.csv'
		w_res = res.tolist();
		w_res.insert(0,flist)
		with open(save_name,'w',newline='') as f:
			writer = csv.writer(f)   
			writer.writerows(w_res)	
	print(f'done {save_name}')
	return save_name

def rdkit2d208_mordred1826(csv_file,s_col,fp,args):
	from rdkit import Chem
	from rdkit.Chem import Descriptors
	from rdkit.ML.Descriptors import MoleculeDescriptors
	#https://github.com/mordred-descriptor/mordred
	from mordred import Calculator, descriptors
	csv_base = csv_file.split('.')[0]
	save_name = f'{fp_feat_dir}/{csv_base}_{fp}.{args.out_format}'  

	df=pd.read_csv(csv_file)	
	smiles_col = df.columns[int(s_col)] if s_col.isdigit() else s_col
	smiles_list= df[smiles_col]
	#mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
	if fp=='rdkit2d':
		mols_gen = (Chem.MolFromSmiles(smi) for smi in smiles_list)
		calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] 
					 for x in Descriptors._descList])
		desc_names = calc.GetDescriptorNames()
		rdkit_descs = [calc.CalcDescriptors(m) for m in mols_gen]
		w_res = rdkit_descs;
		w_res.insert(0,desc_names)
		with open(save_name,'w',newline='') as f:
			writer = csv.writer(f)   
			writer.writerows(w_res)	
			print(f'done {save_name}')
			return save_name
	elif fp=='mordred':
		calc = Calculator(descriptors,ignore_3D=True) #ignore_3D=False)
		mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
		# pandas df
		df = calc.pandas(mols)
		df.to_csv(save_name,index=False)
		print(f'done {save_name}')
		return save_name

def All_Mordred_descriptors(data):
	calc = Calculator(descriptors, ignore_3D=False)
	mols = [Chem.MolFromSmiles(smi) for smi in data]
	# pandas df
	df = calc.pandas(mols)
	return df

def main():
	os.chdir(base)
	parser = argparse.ArgumentParser(description='better to gen bool int float fp in sep fold')	
	  
	parser.add_argument('-f',dest='csv_file', type=str, help="supp only csv") 
	parser.add_argument('-s',dest='smiles_col', type=str, default='smiles', help="default smiles ") 
	parser.add_argument('-fps',dest='fps', nargs='*',default=[],
		help="choice of fps bool int float or empty(default, gen all") 
	parser.add_argument('-threads',dest='threads', type=int,default=20,help="default 20")
	parser.add_argument('-a',dest='action', type=str, default='gen', 
		help="choice gen(gen feat), show_fp_dict, show_fp_list, show_bool_list, show_int_list,show_float_list") 
	parser.add_argument('-out_format', choices=['csv','job'], 
		default='csv', help="csv(default) or job")
	parser.add_argument('-overwrite',action='store_true',default=False, 
		help="overwrite")

	args = parser.parse_args()
	csv_file = args.csv_file;	smiles_col = args.smiles_col
	r_fps = args.fps;action = args.action
	#threads = args.threads;out_format=args.out_format
	#overwrite = args.overwrite
	if action=='show_fp_dict':
		print(descriptor.Extraction().factory);print(fingerprint.Extraction().factory);return
	if action=='show_fp_list':
		print('fingerprint_list=',' '.join(fingerprint_list))
		print('descriptor_list=',' '.join(descriptor_list))
		print('descriptor_other = rdkit2d, mordred')
		return
	if action=='show_bool_list':
		print('fingerprint_list=',' '.join(bool_dtype_list));return
	if action=='show_int_list':
		print('show_int_list=',' '.join(int_dtype_list));return
	if action=='show_float_list':
		print('show_float_list=',' '.join(float_dtype_list));return	
	fps=[]
	if len(r_fps)==0: fps=fingerprint_list+descriptor_list
	if 'bool' in r_fps: fps.extend(bool_dtype_list)	
	if 'int' in r_fps: fps.extend(int_dtype_list)
	if 'float' in r_fps: fps.extend(float_dtype_list)
	for fp in r_fps:
		if fp not in ['bool','int','float']:
			if fp not in fps:
				fps.append(fp)
	print('fps = ', fps)
	#gen_fps(csv_file,smiles_col,fps)
	for fp in fps:
		print('fp = ',fp)
		if fp in ['rdkit2d','mordred']:	
			feat_file = rdkit2d208_mordred1826(csv_file,smiles_col,fp,args)
		else:
			feat_file = gen_one_type_fp(csv_file,smiles_col,fp,args)
		print('done feat_file: {}'.format(feat_file))
	print('all done master,note: the desc fp file might need to be further standardised')

if __name__=='__main__':
	main()


