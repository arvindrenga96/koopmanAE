import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

import torch
import os

#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name, noise = 0.0, theta=2.4,data_version='gulf_mexico_82_87',train_years=3,test_years =1):
    if name == 'pendulum_lin':
        return pendulum_lin(noise)      
    if name == 'pendulum':
        return pendulum(noise, theta)   
    if name=='hydrology':
        return df_hydrology(noise)
    if name == 'sst':
        return sst(data_version)
    else:
        raise ValueError('dataset {} not recognized'.format(name))


def rescale(Xsmall, Xsmall_test):
    #******************************************************************************
    # Rescale data
    #******************************************************************************
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()
    
    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

    return Xsmall, Xsmall_test


def pendulum_lin(noise):
    
    np.random.seed(0)

    def sol(t,theta0):
        S = np.sin(0.5*(theta0) )
        K_S = ellipk(S**2)
        omega_0 = np.sqrt(9.81)
        sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
        theta = 2.0*np.arcsin( S*sn )
        d_sn_du = cn*dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
        return np.stack([theta, d_theta_dt],axis=1)
    
    
    anal_ts = np.arange(0, 2200*0.1, 0.1)
    
    X = sol(anal_ts, 0.8)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
 
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate   
    Xclean = Xclean.T.dot(Q.T)     
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into train and test set 
    X_train = X[0:600]   
    X_test = X[600:]

    X_train_clean = Xclean[0:600]   
    X_test_clean = Xclean[600:]    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, 64, 1






def pendulum(noise, theta=2.4):
    
    np.random.seed(1)

    def sol(t,theta0):
        S = np.sin(0.5*(theta0) )
        K_S = ellipk(S**2)
        omega_0 = np.sqrt(9.81)
        sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
        theta = 2.0*np.arcsin( S*sn )
        d_sn_du = cn*dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
        return np.stack([theta, d_theta_dt],axis=1)
    
    
    anal_ts = np.arange(0, 2200*0.1, 0.1)
    X = sol(anal_ts, theta)
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    Xclean = Xclean.T.dot(Q.T)
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into train and test set 
    X_train = X[0:600]   
    X_test = X[600:]

    X_train_clean = Xclean[0:600]   
    X_test_clean = Xclean[600:]     
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, 64, 1

def load_dataset(file,PREPROCESSED_DIR):
	dataset = np.load(os.path.join(PREPROCESSED_DIR, "{}.npz".format(file)), allow_pickle=True)
	return dataset

def get_data(dataset, index, preprocessed=True,dataset_fold=0):
	unknown =-999    
	data = dataset["data"]
	if preprocessed:
		data = (data-dataset["train_data_means"])/dataset["train_data_stds"]
	data = np.nan_to_num(data, nan=unknown)
	data = data[dataset[index][dataset_fold]]
	return data


def df_hydrology(noise):
    PREPROCESSED_DIR = "../../Fall_22/Benchmarking_caravan/DATA/caravan/PREPROCESSED/camelsgb"
    unknown = -999
    dynamic_channels = [20]#, 21, 22, 23, 24, 25, 26, 27, 28]
    output_channels = [29]
    np.random.seed(1)

    file, index = "train", "train_index"
    dataset = load_dataset(file,PREPROCESSED_DIR)
    data = get_data(dataset, index)
    nodes, window, channels = data.shape
    
    basin =100 
    X = data[basin,:,output_channels+dynamic_channels].T
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    Xclean = Xclean.T.dot(Q.T)
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into train and test set 
    X_train = X[0:2190]   
    X_test = X[2190:]

    X_train_clean = Xclean[0:2190]   
    X_test_clean = Xclean[2190:]     
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, 64, 1


def sst(data_version):

    if data_version == "sst_omri":
        X_train=np.load("/home/kumarv/renga016/Public/sst_dataset_omri/sstday_train.npy")
        X_test=np.load("/home/kumarv/renga016/Public/sst_dataset_omri/sstday_valid.npy")
        t, m, n = X_test.shape
        sst_nanflag = np.full((m, n), True)
        np.save("sst/sst_sst_omri_nanflag.npy",sst_nanflag)
        
        return X_train, X_test,X_train, X_test, m, n
        
    else:
        print('sst/sst_{}.npy'.format(data_version))
        X = np.load('sst/sst_{}.npy'.format(data_version))
    #******************************************************************************
    # Preprocess data
    #******************************************************************************
    t, m, n = X.shape
    print(X.shape)

    #******************************************************************************
    # Slect train data
    #******************************************************************************
    #indices = np.random.permutation(1400)
    indices = range(X.shape[0])
    #training_idx, test_idx = indices[:730], indices[730:1000] 
    # training_idx, test_idx = indices[220:1315], indices[1315:] # 6 years
    #training_idx, test_idx = indices[230:2420], indices[2420:2557] # 6 years
    #training_idx, test_idx = indices[0:1825], indices[1825:2557] # 5 years    
#     training_idx, test_idx = indices[230:1325], indices[1325:210+1325] # 3 years
    training_idx, test_idx = indices[:1825], indices[1825:210+1325] # 3 years
    # training_idx, test_idx = indices[230:1325], indices[1325:2000] # 3 years
    # training_idx, test_idx = indices[210+35*365:210+38*365], indices[210+38*365:210+210+38*365] # 3 years    
    
    # mean subtract
    print(np.min(X))
    X = X.reshape(-1,m*n)
    X -= X.mean(axis=0)   

    
    # scale 
    X = X.reshape(-1,m*n)
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    X = X.reshape(-1,m,n) 
    
    # split into train and test set
    
    X_train = X[training_idx]  
    X_test = X[test_idx]
    
    print(X_train.shape)

 
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test,X_train, X_test, m, n