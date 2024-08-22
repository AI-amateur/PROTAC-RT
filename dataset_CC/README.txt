
1.Intrinsic dataset and External dataset

1).Source
The files "420intrinsic.csv" and "113external.csv" are both sourced from scientific literature.	
	
2).Number
"420intrinsic.csv" contains 420 samples, and "113external.csv" contains 113 samples.

3)spilt
Dataset was split by ten different random (1,2,4,8,16,32,64,128,256,512) seeds to divided into train,validation,test by a proportion of 8:1:1.


2.Incorporative Intrinsic dataset and External dataset

1)incorporation and division
"420intrinsic.csv" and "113external.csv" were incorporated and then divide into The files "457trainN.csv" and "76testN.csv" according to certain conditions.

2).Number
"457trainN.csv" contains 457 samples, and "76testNdata.csv" contains 76 samples.

3).Split
Dataset was split by ten different random (1,2,4,8,16,32,64,128,256,512) seeds to divided into train,validation,test by a proportion of 8:1:1.


2.Test E

The files "6testE.csv" is based on experimentally determined samples, including data for 6 compounds. 

3. Dataset contexts

Each of the aforementioned datasets includes retention time and chromatographic condition information for each sample. 

##If you wish to use them, please extract the chromatographic condition information as a new input variable separately. 

4. Dataset for transfer model

The files "39trainGNN.csv" is the training set for the transfer model, containing 39 samples with consistent conditions.



