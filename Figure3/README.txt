*All the results were the average value of ten runs. In each run, the 420intrinsic dataset was split randomly as training, validation and test dataset (ratio: 0.8, 0.1, 0.1) using the random seed chosen from 1, 2, 4, 8, 16, 32, 64, 128, 256 and 512.


1. The file "4ml_reg.csv" records the predictive performance of four machine learning algorithms��XGBoostoost,RF,KNN,and SVM��based on 12	molecular fingerprints and 12 molecular descriptors;The file "4ml_reg_average.csv" records the average predictive performance for each fingerprint and descriptor.

2. The file "fcnn_reg.csv" records the predictive performance of a deep learning algorithms��FCNN��based on 12 molecular fingerprints and 12 molecular descriptors.The file "fcnn_reg_average.csv" records the average predictive performance for each fingerprint and descriptor.



2. The files "R2.csv" and "RMSE.csv" respectively contain the data required for plotting violin plots of R-squared R2 and RMSE for five models.


3. The file "5_model_violin.html" records the code for plotting violin plots.