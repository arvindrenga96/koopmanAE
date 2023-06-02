import torch
import copy
from torch import nn
import numpy as np

from tools import *



def train_sst(model, train_loader, val_loader, lr, weight_decay,
          lamb, num_epochs, learning_rate_change, epoch_update, data_version, device,
          nu=0.0, eta=0.0, backward=0, steps=1, steps_back=1, gradclip=1, earlystopping=True, patience_max=40):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = device
             
            
    def lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
                    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
                    if epoch in decayEpoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_decay_rate
                        return optimizer
                    else:
                        return optimizer
                        
                     
        

    criterion = nn.MSELoss().to(device)


    epoch_hist = []
    loss_hist = []
    epoch_loss = []
    patience =0
    min_val_loss = 100000

    nanflag = np.load('sst/sst_all_years_{}.npz'.format(data_version))['nanset']
    # nanflag = np.load('sst/sst_{}_nanflag.npy'.format(data_version))

    for epoch in range(num_epochs):
        #print(epoch)
        for batch_idx, data_list in enumerate(train_loader):
            model.train()
            out, out_back = model(data_list[0].to(device), mode='forward')

            for k in range(steps):
                if k == 0:
                    loss_fwd = criterion(out[k][:,:,nanflag], data_list[k+1][:,:,nanflag].to(device))
                else:
                    loss_fwd += criterion(out[k][:,:,nanflag], data_list[k+1][:,:,nanflag].to(device))

            
            loss_identity = criterion(out[-1][:,:,nanflag], data_list[0][:,:,nanflag].to(device)) * steps


            loss_bwd = 0.0
            loss_consist = 0.0


            if backward == 1:
                out, out_back = model(data_list[-1].to(device), mode='backward')
   

                for k in range(steps_back):
                    
                    if k == 0:
                        loss_bwd = criterion(out_back[k][:,:,nanflag], data_list[::-1][k+1][:,:,nanflag].to(device))
                    else:
                        loss_bwd += criterion(out_back[k][:,:,nanflag], data_list[::-1][k+1][:,:,nanflag].to(device))
                        
                if eta!=0:               
                    A = model.dynamics.dynamics.weight
                    B = model.backdynamics.dynamics.weight

                    K = A.shape[-1]

                    for k in range(1,K+1):
                        As1 = A[:,:k]
                        Bs1 = B[:k,:]
                        As2 = A[:k,:]
                        Bs2 = B[:,:k]

                        Ik = torch.eye(k).float().to(device)

                        if k == 1:
                            loss_consist = (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                             torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
                        else:
                            loss_consist += (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                                             torch.sum((torch.mm(As2, Bs2)-  Ik)**2) ) / (2.0*k)





    #                Ik = torch.eye(K).float().to(device)
    #                loss_consist = (torch.sum( (torch.mm(A, B)-Ik )**2)**1 + \
    #                                         torch.sum( (torch.mm(B, A)-Ik)**2)**1 )
    #   


                else:
                    loss_consist=0
            loss = loss_fwd + lamb * loss_identity +  nu * loss_bwd + eta * loss_consist

                

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip) # gradient clip
            optimizer.step()           

        # schedule learning rate decay


        lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)
        loss_hist.append(loss)                
        epoch_loss.append(epoch)
        
        
        if (epoch) % 1 == 0:
                print('********** Epoche %s **********' %(epoch+1))
                
                print("loss identity: ", loss_identity.item())
                if backward == 1 and eta!=0:
                    print("loss backward: ", loss_bwd.item())
                    print("loss consistent: ", loss_consist.item())
                elif backward == 1:
                    print("loss backward: ", loss_bwd.item())
                print("loss forward: ", loss_fwd.item())
                print("loss sum: ", loss.item())

                for _, data in enumerate(val_loader):
                    model.eval()
                    error_temp = []
                    for i in range(30):
                        z = model.encoder(data[i].to(device))  # embedd data in latent space
                        for j in range(175):
                                z = model.dynamics(z.squeeze(1))
                                # print(z.shape)
                                x_pred = model.decoder(z.unsqueeze(1))  # map back to high-dimensional space

                                target_temp = data[i+1 + j].unsqueeze(1).numpy()

                                error_temp.append(np.linalg.norm(x_pred.detach().cpu().numpy()[:,:,nanflag] - target_temp[:,:,nanflag]) / np.linalg.norm(target_temp[:,:,nanflag]))

                    val_loss =np.mean(error_temp)
                    print (f"Validation Loss is {val_loss}")

                if val_loss<=min_val_loss:
                    min_val_loss = val_loss
                    patience = 0
                    best_model = copy.deepcopy(model)
                else:
                    patience +=1

                if patience>patience_max and earlystopping:
                    print("Early stopping")
                    break

                epoch_hist.append(epoch+1)
            #
            # if (model.__class__.__name__ != "ConvLSTM" and model    .__class__.__name__ != "LSTM"):
            #     if hasattr(model.dynamics, 'dynamics'):
            #         w, _ = np.linalg.eig(model.dynamics.dynamics.weight.data.cpu().numpy())
            #         print(np.abs(w))


    if backward == 1 and eta!=0:
        loss_consist = loss_consist.item()
                
                
    return best_model, optimizer, [epoch_hist, loss_fwd.item(), loss_consist]
