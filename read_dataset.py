import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

import torch
import os
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name, save_folder, noise = 0.0, theta=2.4,data_version='gulf_mexico_82_87',train_years=3,start_year=0):
    if name == 'pendulum_lin':
        return pendulum_lin(noise)      
    if name == 'pendulum':
        return pendulum(noise,save_folder, theta)
    if name=='hydrology':
        return df_hydrology(noise)
    if name == 'sst':
        return sst(data_version,train_years,start_year,save_folder)
    if name == 'lorenz':
        return lorenz(noise,save_folder)
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


def lorenz(noise, save_folder):
    """Generates train and test sets of noisy and clean Lorenz attractor time series.

    Args:
        noise (float): Noise level for noisy time series.

    Returns:
        X_train (ndarray): Training set of noisy time series.
        X_test (ndarray): Test set of noisy time series.
        X_train_clean (ndarray): Training set of clean time series.
        X_test_clean (ndarray): Test set of clean time series.
        64 (int): Dimensionality of time series.
        1 (int): Number of time series.
    """
    # Lorenz Attractor parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # Lorenz System of differential equations
    def lorenz_system(current_state, t):
        x, y, z = current_state
        return [
            sigma * (y - x),  # dx/dt
            x * (rho - z) - y,  # dy/dt
            x * y - beta * z,  # dz/dt
        ]

    # Initial state for the system
    initial_state = [1.0, 1.0, 1.0]

    # Time points
    t = np.arange(0.0, 50.0, 0.01)

    # Solve Lorenz system equations
    solution = odeint(lorenz_system, initial_state, t)

    # Generate noisy and clean versions of solution
    X = solution.T + np.random.standard_normal(solution.T.shape) * noise  # Noisy
    X_clean = solution.T.copy()  # Clean



    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64, 3))
    Q, _ = np.linalg.qr(Q)  # QR decomposition for orthonormality

    np.savez(save_folder + '/data.npz', Q=Q, X=X, X_clean=X_clean)

    X = X.T @ Q.T  # Rotate X
    X_clean = X_clean.T @ Q.T  # Rotate X_clean

    # Scale data to [-1, 1] and save the original min and range
    min_X, min_X_clean = np.min(X), np.min(X_clean)
    range_X, range_X_clean = np.ptp(X), np.ptp(X_clean)

    X = 2 * (X - min_X) / range_X - 1
    X_clean = 2 * (X_clean - min_X_clean) / range_X_clean - 1

    # Split into train and test sets
    X_train, X_test = np.split(X, [2000])
    X_train_clean, X_test_clean = np.split(X_clean, [2000])


    return X_train, X_test, X_train_clean, X_test_clean, 64, 1


def pendulum(noise,save_folder, theta=2.4):
    
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
    
    
    anal_ts = np.arange(0, 4000*0.1, 0.1)
    X = sol(anal_ts, theta)
    print (f'Shape of dataset is {X.shape}')
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,2))
    Q,_ = np.linalg.qr(Q)

    np.savez(save_folder + '/data.npz', Q=Q, X=X, X_clean=Xclean)

    X = X.T.dot(Q.T) # rotate
    Xclean = Xclean.T.dot(Q.T)
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    train_split = 400
    test_split = 1900
    # split into train and test set 
    X_train = X[0:train_split]
    print (f'Shape of X_train dataset is {X_train.shape}')

    X_test = X[test_split:]
    print (f'Shape of X_test dataset is {X_test.shape}')


    X_train_clean = Xclean[0:train_split]
    X_test_clean = Xclean[test_split:]
    
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
    unknown = -999
    dynamic_channels = [20]#, 21, 22, 23, 24, 25, 26, 27, 28]
    output_channels = [29]
    np.random.seed(1)

    X=np.load("/home/kumarv/renga016/Public/DATA/caravan_camels_gb/RAW/data.npy")
    
    print("Xshape",X.shape)
    
    
    basin =100 
    X = X[basin,:,output_channels].T
    print("Number of NANs",np.count_nonzero(np.isnan(X)))

    
    
    X = X.T
    Xclean = X.copy()
    X += np.random.standard_normal(X.shape) * noise
    
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((64,1))
    Q,_ = np.linalg.qr(Q)
    
    X = X.T.dot(Q.T) # rotate
    Xclean = Xclean.T.dot(Q.T)
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into train and test set 
    X_train = X[0:3650*3]   
    X_test = X[3650*3:]

    X_train_clean = Xclean[0:3650*3]   
    X_test_clean = Xclean[3650*3:]    
    
    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, 64, 1


import numpy as p


def sst(data_version,train_years, start_year, save_folder):
    """
    Load and preprocess SST dataset.

    Args:
        data_version (str): The version of the SST dataset to load.

    Returns:
        tuple: A tuple containing the preprocessed train and test sets, along with other dimensions.

    """

    if data_version == "sst_omri":
        # Load SST dataset for 'sst_omri' version
        X_train = np.load("/home/kumarv/renga016/Public/sst_dataset_omri/sstday_train.npy")
        X_test = np.load("/home/kumarv/renga016/Public/sst_dataset_omri/sstday_valid.npy")
        t, m, n = X_test.shape

        # Create nanflag for 'sst_omri'
        sst_nanflag = np.full((m, n), True)
        np.save("sst/sst_sst_omri_nanflag.npy", sst_nanflag)

        # Return the loaded data
        return X_train, X_test, X_train, X_test, m, n
    else:
        # Load SST dataset for other versions

        # print('sst/sst_{}.npy'.format(data_version))
        # X = np.load('sst/sst_{}.npy'.format(data_version))
        print('sst/sst_all_years_{}.npz'.format(data_version))
        X = np.load('sst/sst_all_years_{}.npz'.format(data_version))['dataset']



    # Preprocess data
    t, m, n = X.shape
    print(X.shape)

    start_index = int(start_year*365) + 220
    train_end_index = start_index + int(train_years*365)

    # Select train data
    indices = range(X.shape[0])
    training_idx, val_idx, test_idx = indices[start_index:train_end_index], indices[train_end_index:train_end_index + 210], indices[train_end_index + 365: train_end_index + 365 + 210] 
    # training_idx, val_idx, test_idx = indices[220:1315], indices[1315:210 + 1315], indices[1315+365:210 + 1315+365]  # 3 years

    # Reshape
    X = X.reshape(-1, m * n)
    print (f"Shape of X is {X.shape}") # no_of_timesteps, 100, 180

    # Save original mean, min and ptp
    original_mean = X[start_index:train_end_index + 365 + 210].mean(axis=0)

    # Mean subtract
    X -= original_mean

    original_min = np.min(X[start_index:train_end_index + 365 + 210])
    original_ptp = np.ptp(X[start_index:train_end_index + 365 + 210])

    # Scale
    X = 2 * (X - original_min) / original_ptp - 1
    print(np.min(X))
    print(np.max(X))
    print(np.mean(X))

    # Reshape
    X = X.reshape(-1, m, n)


    # Split into train and test set
    X_train = X[training_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    avg_prediction = np.zeros(X_test.shape)

    for index, value in enumerate(test_idx):
        # Calculate the indices

        # indices = [value - 365, value - 730, value - 1095]
        indices = [value-(idx+2)*365 for idx in range(train_years)]

        # Take the average along the first axis (time axis)
        averages = np.mean(X[indices], axis=0)

        # Print the shape of the resulting average array
        avg_prediction[index] = averages

    np.savez(save_folder + '/data.npz', Avg_Prediction=avg_prediction, mean=original_mean, min=original_min, ptp=original_ptp)


    # Return train and test set
    return X_train, X_val, X_test, X_train, X_val, X_test, m, n
