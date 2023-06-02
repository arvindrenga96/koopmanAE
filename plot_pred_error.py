import time
import argparse
import sys
import numpy as np

from scipy import stats
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.available
mpl.style.use('seaborn-paper') 


from matplotlib.ticker import StrMethodFormatter


# result1 = np.load('results_pendulum/000_pred.npy')                  
# result2 = np.load('results_back_pendulum/000_pred.npy')   
# result3 = np.load('results_pendulum_inn/000_pred.npy')





# result1 = np.load('results_pendulum_6_noise_03_seed_3_dae/000_pred.npy')                  
# result2 = np.load('results_pendulum_6_noise_03_seed_3_kae/000_pred.npy')   
# result3 = np.load('results_pendulum_6_noise_03_seed_3_kae_inn/000_pred.npy')
# result4 = np.load('results_pendulum_6_noise_03_seed_3_lstm/000_pred.npy')


# result1 = np.load('results_det_hydrology_6_noise_03/000_pred.npy')                  
# result2 = np.load('results_det_back_hydrology_6_noise_03/000_pred.npy')   
# result3 = np.load('results_det_back_hydrology_6_noise_03_inn/000_pred.npy')



if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Please provide the All results folder name as command-line arguments.')
    else:
        result1 = np.load(sys.argv[1])
        result2 = np.load(sys.argv[2])
        result3 = np.load(sys.argv[3])
        result4 = np.load(sys.argv[4])
        save_path = sys.argv[5]


def moving_average(a, n=4) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n    
    

# fig = plt.figure(figsize=(12,4))
#
# # Bright colors
# bright_color1 = (44/255, 117/255, 255/255)  # Electric Blue
# bright_color2 = (255/255, 128/255, 0/255)  # Vibrant Orange
#
# # Dull colors
# dull_color1 = (115/255, 160/255, 115/255)  # Muted Green
# dull_color2 = (209/255, 146/255, 144/255)  # Dusty Rose
#
# # plt.plot(np.mean(result1, axis=0), '-', lw=2, label='Koopman AE', color='#377eb8')
# # plt.fill_between(x=range(result1.shape[1]), y1=np.mean(result1, axis=0)-np.var(result1, axis=0)**0.5, y2=np.mean(result1, axis=0)+np.var(result1, axis=0)**0.5, color='#377eb8', alpha=0.2)
# #
# # plt.plot(np.mean(result2, axis=0), '-', lw=2, label='Consistent Koopman AE', color='#e41a1c')
# # plt.fill_between(x=range(result2.shape[1]), y1=np.mean(result2, axis=0)-np.var(result2, axis=0)**0.5, y2=np.mean(result2, axis=0)+np.var(result2, axis=0)**0.5, color='#e41a1c', alpha=0.2)
# #
# # plt.plot(np.mean(result3, axis=0), '-', lw=2, label='Koopman AE INN',color='#1ae472')
# # plt.fill_between(x=range(result3.shape[1]), y1=np.mean(result3, axis=0)-np.var(result3, axis=0)**0.5, y2=np.mean(result3, axis=0)+np.var(result3, axis=0)**0.5, color='#1ae472', alpha=0.2)
# #
# # plt.plot(np.mean(result4, axis=0), '-', lw=2, label='LSTM',color='#A020F0')
# # plt.fill_between(x=range(result4.shape[1]), y1=np.mean(result4, axis=0)-np.var(result4, axis=0)**0.5, y2=np.mean(result4, axis=0)+np.var(result4, axis=0)**0.5, color='#A020F0', alpha=0.2)
#
# plt.plot(np.mean(result1, axis=0), '-', lw=2, label='KAE', color=bright_color1)
# plt.fill_between(x=range(result1.shape[1]), y1=np.mean(result1, axis=0) - np.var(result1, axis=0) ** 0.5,
#                  y2=np.mean(result1, axis=0) + np.var(result1, axis=0) ** 0.5, color='#377eb8', alpha=0.2)
#
# plt.plot(np.mean(result2, axis=0), '-', lw=2, label='C-KAE', color=bright_color2)
# plt.fill_between(x=range(result2.shape[1]), y1=np.mean(result2, axis=0) - np.var(result2, axis=0) ** 0.5,
#                  y2=np.mean(result2, axis=0) + np.var(result2, axis=0) ** 0.5, color='#e41a1c', alpha=0.2)
#
# plt.plot(np.mean(result4, axis=0), '-', lw=2, label='LSTM', color=dull_color2)
# plt.fill_between(x=range(result4.shape[1]), y1=np.mean(result4, axis=0) - np.var(result4, axis=0) ** 0.5,
#                  y2=np.mean(result4, axis=0) + np.var(result4, axis=0) ** 0.5, color='#A020F0', alpha=0.2)
#
# plt.plot(np.mean(result3, axis=0), '-', lw=2, label='KIA', color='#e41a1c')
# plt.fill_between(x=range(result3.shape[1]), y1=np.mean(result3, axis=0) - np.var(result3, axis=0) ** 0.5,
#                  y2=np.mean(result3, axis=0) + np.var(result3, axis=0) ** 0.5, color='#1ae472', alpha=0.2)
#
#
#
#
# plt.tick_params(axis='x', labelsize=18)
# plt.tick_params(axis='y', labelsize=18)
# plt.tick_params(axis='both', which='minor', labelsize=16)
# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
#
# plt.xlabel('time, t',fontsize=18)
# plt.ylabel('Prediction error',fontsize=18)
#
# plt.grid(False)
# maxmax = np.maximum(result1.max(), result3.max())
# plt.legend(fontsize=18, loc="upper left")
# fig.tight_layout()
# plt.savefig(f"{save_path}/plot")
# plt.show()


# Bright colors
bright_color1 = (44/255, 117/255, 255/255)  # Electric Blue
bright_color2 = (255/255, 128/255, 0/255)  # Vibrant Orange

# Dull colors
dull_color1 = (115/255, 160/255, 115/255)  # Muted Green
dull_color2 = (209/255, 146/255, 144/255)  # Dusty Rose

fig, ax = plt.subplots(figsize=(12,6))

# Assigning color for each result
color_map = {
    "result1": bright_color1,
    "result2": bright_color2,
    "result3": '#e41a1c',
    "result4": '#008000',
}

# Results and their labels
results = [result1, result2, result3, result4]
labels = ["KAE", "C-KAE", "KIA", "LSTM"]

# Plotting
for result, label, color in zip(results, labels, color_map.values()):
    mean_result = np.mean(result, axis=0)
    var_result = np.var(result, axis=0)**0.5
    x_values = range(result.shape[1])

    ax.plot(mean_result, '-', lw=2, label=label, color=color)
    ax.fill_between(x=x_values, y1=mean_result - var_result, y2=mean_result + var_result, color=color, alpha=0.2)

plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
# Setting x and y labels with font size
ax.set_xlabel('Timestep -->', fontsize=30)
ax.set_ylabel('Prediction Error', fontsize=30)

# Setting gri
ax.grid(True, linestyle='--', alpha=0.6)

# Adding a black box around the plot
for spine in ax.spines.values():
    spine.set_linewidth(3)
    spine.set_edgecolor('black')

# Setting legend inside the plot in the best location determined by matplotlib
ax.legend(loc='upper left', fontsize=30)

# Applying tight layout
fig.tight_layout()

# Saving the figure
plt.savefig(f"{save_path}/plot", bbox_inches='tight')

# Displaying the plot
plt.show()
