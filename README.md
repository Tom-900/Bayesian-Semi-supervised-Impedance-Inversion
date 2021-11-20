# Semi-supervised Impedance Inversion by Bayesian Neural Network Based on 2-d CNN Pre-training
This code is written for the manuscript: Semi-supervised Impedance Inversion by Bayesian Neural Network Based on 2-d CNN Pre-training. 

There are three folders in the project. In "data" part, there is 50ETF option and stock option data collected from Chinese stock and fund market. The data is divided into training set and test set. "test_data_BS.py" is specially used for B-S model. There is also a folder named "torch-data", which saves the torch tensors created by "data_process.py".

In "core" part, the 

In "main" part, we could apply the models and the data to predict option price, and compare their strengths and weaknesses through five measurements: MSE, RMSE, MAP, MAPE and PCC.
