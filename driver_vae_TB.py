import time
import argparse

import numpy as np

from scipy import stats
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.available
mpl.style.use('seaborn-paper')
from model_vae import DynamicVAE
from train_vae_TB import train_vae
import torch
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable, Function

from timeit import default_timer as timer

import torch.nn.init as init

from read_dataset import data_from_name
from model import *
from tools import *

from train import *

# import ristretto as ro
# from ristretto.svd import compute_rsvd

import os

from scipy import stats

from Adam_new import *


# python driver.py --model net --dataset harmonic --epochs 400 --batch 32 --folder results --lamb 4.0 --steps 5


class shallow_re(nn.Module):
    def __init__(self, m, n, b):
        super(shallow_re, self).__init__()
        self.decoder = decoderNet(m, n, b)

    def forward(self, x):
        out = self.decoder(x)
        return out


# ==============================================================================
# Training settings
# ==============================================================================
parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--model', type=str, default='net', metavar='N', help='Model')
#
parser.add_argument('--dataset', type=str, default='harmonic', metavar='N', help='dataset')
#
parser.add_argument('--lr', type=float, default=1e-2, metavar='N', help='learning rate (default: 0.01)')
#
parser.add_argument('--wd', type=float, default=0.0, metavar='N', help='weight_decay (default: 1e-5)')
#
parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--batch', type=int, default=64, metavar='N', help='batch size (default: 10000)')
#
parser.add_argument('--batch_test', type=int, default=50, metavar='N', help='batch size  for test set (default: 10000)')
#
parser.add_argument('--plotting', type=bool, default=True, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--folder', type=str, default='results_back', help='specify directory to print results to')
#
parser.add_argument('--lamb', type=float, default='4', help='PCL penalty lambda hyperparameter')
#
parser.add_argument('--eta', type=float, default='1e-3', help='Depricated')
#
parser.add_argument('--steps', type=int, default='3', help='steps for omega')
#
parser.add_argument('--bottleneck', type=int, default='2', help='bottleneck')
#
parser.add_argument('--lr_update', type=int, nargs='+', default=[100, 300, 500],
                    help='Decrease learning rate at these epochs.')
#
parser.add_argument('--lr_decay', type=float, default='0.2', help='PCL penalty lambda hyperparameter')
#
parser.add_argument('--pred_steps', type=int, default='1000', help='Prediction steps')

parser.add_argument('--backward', type=int, default=1, help='whether to train also with backward dynamics')
#
parser.add_argument('--seed', type=int, default='1', help='Prediction steps')
#


args = parser.parse_args()



set_seed()
device = get_device()


# ******************************************************************************
# Create folder to save results
# ******************************************************************************
if not os.path.isdir(args.folder):
    os.mkdir(args.folder)

# ******************************************************************************
# load data
# ******************************************************************************
Xtrain, Xtest, m, n = data_from_name(args.dataset)
Xfull = np.concatenate((Xtrain, Xtest))

# ******************************************************************************
# Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
# ******************************************************************************
Xtrain = add_channels(Xtrain)
Xtest = add_channels(Xtest)
Xfull = add_channels(Xfull)

# transfer to tensor
Xtrain = torch.from_numpy(Xtrain).float().contiguous()
Xtest = torch.from_numpy(Xtest).float().contiguous()
Xfull = torch.from_numpy(Xfull).float().contiguous()

# ******************************************************************************
# Create Dataloader objects
# ******************************************************************************


trainDat = []
start = 0
for i in np.arange(args.steps, -1, -1):
    if i == 0:
        trainDat.append(Xtrain[start:].float())
    else:
        trainDat.append(Xtrain[start:-i].float())
    start += 1

train_data = torch.utils.data.TensorDataset(*trainDat)
train_loader = DataLoader(dataset=train_data,
                          batch_size=args.batch,
                          shuffle=True)

testDat = []
start = 0
for i in np.arange(args.steps, -1, -1):
    if i == 0:
        testDat.append(Xtest[start:].float())
    else:
        testDat.append(Xtest[start:-i].float())
    start += 1

test_data = torch.utils.data.TensorDataset(*testDat)
test_loader = DataLoader(dataset=test_data,
                         batch_size=args.batch_test,
                         shuffle=False)

del (trainDat, testDat)

# ==============================================================================
# Model
# ==============================================================================
print(Xtrain.shape)
if (args.model == 'net'):
    model = DynamicVAE(Xtrain.shape[2], Xtrain.shape[3], args.bottleneck, args.steps)
    model.apply(weights_init)
    #model.decoder.fc3logvar.weight.
    print('net')

# elif(args.model == 'big'):
#    model = big_autoencoder(Xtrain.shape[2], Xtrain.shape[3])
#    model.apply(weights_init)
#    print('big')

#model = torch.nn.DataParallel(model).to(device)
model = model.to(device)

# ==============================================================================
# Model summary
# ==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
print('************')
print(model)

# ==============================================================================
# Start training
# ==============================================================================
model, optimizer, epoch_hist = train_vae(model, train_loader, test_loader,
                                                              lr=args.lr, weight_decay=args.wd, lamb=args.lamb,
                                                              num_epochs=args.epochs,
                                                              learning_rate_change=args.lr_decay,
                                                              epoch_update=args.lr_update,
                                                              eta=args.eta, backward=args.backward)



# with open(args.folder+"/model.pkl", "wb") as f:
#    torch.save(model,f)
torch.save(model.state_dict(), args.folder + '/model' + '.pkl')

for param_group in optimizer.param_groups:
    print(param_group['weight_decay'])
    print(param_group['weight_decay_adapt'])


# ******************************************************************************
# Prediction
# ******************************************************************************
Xinput, Xtarget = Xtest[:-1], Xtest[1:]

error = []
for i in range(30):
    error_temp = []

    mu, logvar = model.encoder(Xinput[i].float().to(device))  # embedd data in latent space
    var = torch.exp(logvar)
    for j in range(args.pred_steps):
        mu, cholesk_dec = model.dynamics.forward_dist(mu, var, j + 1)
        cholesk_dec = cholesk_dec.unsqueeze(0)
        z = model.reparametrize_multidim(mu, cholesk_dec)
        x_pred = model.decoder.forward(z)
        target_temp = Xtarget[i + j].data.cpu().numpy().reshape(m, n)
        error_temp.append(
            np.linalg.norm(x_pred.data.cpu().numpy().reshape(m, n) - target_temp) / np.linalg.norm(target_temp))

    error.append(np.asarray(error_temp))
error = np.asarray(error)

fig = plt.figure(figsize=(15, 12))
plt.plot(error.mean(axis=0), 'o--', lw=3, label='', color='#377eb8')
plt.fill_between(x=range(error.shape[1]), y1=np.quantile(error, .05, axis=0), y2=np.quantile(error, .95, axis=0),
                 color='#377eb8', alpha=0.2)

plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=22)
plt.locator_params(axis='y', nbins=10)
plt.locator_params(axis='x', nbins=10)

plt.ylabel('Relative prediction error', fontsize=22)
plt.xlabel('Time step', fontsize=22)
plt.grid(False)
# plt.yscale("log")
plt.ylim([0.0, error.max() * 2])
# plt.legend(fontsize=22)
fig.tight_layout()
plt.savefig(args.folder + '/000prediction' + '.png')
# plt.savefig(args.folder +'/000prediction' +'.eps')

plt.close()

np.save(args.folder + '/000_pred.npy', error)

## ******************************************************************************
## Empedding
## ******************************************************************************
#Xinput, Xtarget = Xtest[:-1], Xtest[1:]
#
#emb = []
#mu_emb = []
#var_emb = []
#
#mu, logvar = model.encoder(Xinput[0].float().to(device))  # embedd data in latent space
#var = torch.exp(logvar)
#
#for j in range(args.pred_steps):
#    mu, cholesk_dec = model.dynamics.forward_dist(mu, var, j + 1)
#    cholesk_dec = cholesk_dec.unsqueeze(0)
#    z = model.reparametrize_multidim(mu, cholesk_dec)
#    emb.append(z.data.cpu().numpy().reshape(args.bottleneck))
#    mu_emb.append(mu.data.cpu().numpy().reshape(args.bottleneck))
#    cholesk_dec_numpy = cholesk_dec.data.cpu().numpy().reshape((args.bottleneck, args.bottleneck))
#    var_emb.append(np.matmul(cholesk_dec_numpy, cholesk_dec_numpy.T))
#
#emb = np.asarray(emb)
#mu_emb = np.asarray(mu_emb)
#var_emb = np.asarray(var_emb)
#fig = plt.figure(figsize=(15, 15))
#ax = fig.add_subplot(111)
#plt.plot(emb[:, 0], emb[:, 1], '-', lw=1, label='', color='#377eb8')
#plt.plot(mu_emb[:, 0], mu_emb[:, 1], '-', lw=1, label='', color='#377eb8')
#for mu_i, cov_i in zip(mu_emb, var_emb):
#    confidence_ellipse(mu_i, cov_i, ax, n_std=3, alpha=0.2, facecolor='blue')
#plt.xlim(-1.6, 1.6)
#plt.ylim(-1.6, 1.6)
#
#plt.tick_params(axis='x', labelsize=22)
#plt.tick_params(axis='y', labelsize=22)
#plt.locator_params(axis='y', nbins=10)
#plt.locator_params(axis='x', nbins=10)
#
#plt.ylabel('y', fontsize=22)
#plt.xlabel('x', fontsize=22)
#plt.grid(False)
## plt.yscale("log")
## plt.legend(fontsize=22)
#fig.tight_layout()
#plt.savefig(args.folder + '/embedding' + '.png')
## plt.savefig(args.folder +'/000prediction' +'.eps')
#
#plt.close()

# ******************************************************************************
# Eigenvalues
# ******************************************************************************
model.eval()
A = model.dynamics.dynamics.weight.cpu().data.numpy()
# A =  model.module.test.data.cpu().data.numpy()
w, v = np.linalg.eig(A)
print(np.abs(w))

fig = plt.figure(figsize=(6.1, 6.1), facecolor="white", edgecolor='k', dpi=150)
plt.scatter(w.real, w.imag, c='#dd1c77', marker='o', s=15 * 6, zorder=2, label='Eigenvalues')

maxeig = 1.4
plt.xlim([-maxeig, maxeig])
plt.ylim([-maxeig, maxeig])
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)

plt.xlabel('Real', fontsize=22)
plt.ylabel('Imaginary', fontsize=22)
plt.tick_params(axis='y', labelsize=22)
plt.tick_params(axis='x', labelsize=22)
plt.axhline(y=0, color='#636363', ls='-', lw=3, zorder=1)
plt.axvline(x=0, color='#636363', ls='-', lw=3, zorder=1)

# plt.legend(loc="upper left", fontsize=16)
t = np.linspace(0, np.pi * 2, 100)
plt.plot(np.cos(t), np.sin(t), ls='-', lw=3, c='#636363', zorder=1)
plt.tight_layout()
plt.show()
plt.savefig(args.folder + '/000eigs' + '.png')
plt.savefig(args.folder + '/000eigs' + '.eps')
plt.close()