from torch import nn
import torch
from INN.invertible_net import main_file as inn
import numpy as np
import random





# def set_seed(seed_value):
#     torch.manual_seed(seed_value)
#     torch.cuda.manual_seed_all(seed_value)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed_value)
#     random.seed(seed_value)


def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega


class encoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(encoderNet, self).__init__()
        self.N = m * n
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.N, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, b)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))        
        x = self.fc3(x)
        
        return x


class decoderNet(nn.Module):
    def __init__(self, m, n, b, ALPHA = 1):
        super(decoderNet, self).__init__()

        self.m = m
        self.n = n
        self.b = b

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(b, 16*ALPHA)
        self.fc2 = nn.Linear(16*ALPHA, 16*ALPHA)
        self.fc3 = nn.Linear(16*ALPHA, m*n)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        # print(x[0].shape)
        x = x.view(-1, 1, self.b)
        x = self.tanh(self.fc1(x)) 
        x = self.tanh(self.fc2(x)) 
        x = self.tanh(self.fc3(x))
        x = x.view(-1, 1, self.m, self.n)
        return x



class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

        
    def forward(self, x):
        x = self.dynamics(x)
        return x


class dynamics_back(nn.Module):
    def __init__(self, b, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())     

    def forward(self, x):
        x = self.dynamics(x)
        return x




class koopmanAE(nn.Module):
    def __init__(self, m, n, b, steps, steps_back, alpha = 1, init_scale=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        
        self.encoder = encoderNet(m, n, b, ALPHA = alpha)
        self.dynamics = dynamics(b, init_scale)
        self.backdynamics = dynamics_back(b, self.dynamics)
        self.decoder = decoderNet(m, n, b, ALPHA = alpha)


    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()

        
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))

            out.append(self.decoder(z.contiguous())) 
            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(self.decoder(q))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back
        
    
# class ConditionedAffineCoupling(nn.Module):
#     def __init__(self, input_dim, condition_dim, hidden_dim):
#         super(ConditionedAffineCoupling, self).__init__()
#         self.input_dim = input_dim
#         self.condition_dim = condition_dim
#         self.hidden_dim = hidden_dim

#         self.scale1 = nn.Sequential(
#             nn.Linear(condition_dim, hidden_dim,bias=False),
#             # nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim // 2,bias=False),
#             # nn.Tanh()
#         )
#         self.shift1 = nn.Sequential(
#             nn.Linear(condition_dim, hidden_dim,bias=False),
#             # nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim // 2,bias=False),
#         )
#         self.scale2 = nn.Sequential(
#             nn.Linear(condition_dim, hidden_dim,bias=False),
#             # nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim // 2,bias=False),
#             # nn.Tanh()
#         )
#         self.shift2 = nn.Sequential(
#             nn.Linear(condition_dim, hidden_dim,bias=False),
#             # nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim // 2,bias=False),
#         )

#     def forward(self, x, condition, reverse=False):
#         x1, x2 = x[:, :self.input_dim // 2], x[:, self.input_dim // 2:]

#         if not reverse:
#             s1 = self.scale1(condition)
#             t1 = self.shift1(condition)

#             s2 = self.scale2(condition)
#             t2 = self.shift2(condition)

            
#             # print("x1", x1.shape)
#             # print("t1", t1.shape)
#             y1 = x1 * torch.exp(s1) + t1
#             y2 = x2 * torch.exp(s2) + t2

#             y = torch.cat([y1, y2], dim=1)
#         else:
#             s1 = self.scale1(condition)
#             t1 = self.shift1(condition)

#             s2 = self.scale2(condition)
#             t2 = self.shift2(condition)

#             x1 = (x1 - t1) * torch.exp(-s1)
#             x2 = (x2 - t2) * torch.exp(-s2)
#             y = torch.cat([x1, x2], dim=1)

#         return y

# class ConditionalINN(nn.Module):
#     def __init__(self, input_dim, condition_dim, hidden_dim, n_blocks):
#         super(ConditionalINN, self).__init__()
#         self.input_dim = input_dim
#         self.condition_dim = condition_dim
#         self.hidden_dim = hidden_dim
#         self.n_blocks = n_blocks

#         self.layers = nn.ModuleList([
#             ConditionedAffineCoupling(input_dim, condition_dim, hidden_dim)
#             for _ in range(n_blocks)
#         ])

#     def forward(self, x, condition, reverse=False):
#         if not reverse:
#             for layer in self.layers:
#                 x = layer(x, condition)
#         else:
#             for layer in reversed(self.layers):
#                 x = layer(x, condition, reverse=True)
#         return x
    
# class koopmanAE_INN(nn.Module):
#     def __init__(self, m, n, b, steps, steps_back, alpha = 1, init_scale=1):
#         super(koopmanAE_INN, self).__init__()
#         self.steps = steps
#         self.steps_back = steps_back
#         self.encoder = encoderNet(m, n, b, ALPHA = alpha)
#         self.decoder = decoderNet(m, n, b, ALPHA = alpha)
#         # self.inn_fc  =  lambda input_data, output_data: torch.nn.Sequential(
#         #                              torch.nn.Linear(input_data, input_data, bias=False),
#         #                              torch.nn.ReLU(),
#         #                              torch.nn.Linear(output_data, output_data, bias=False))
#         # self.inn_fc  =  lambda input_data, output_data: torch.nn.Sequential(
#         #                              torch.nn.Linear(input_data, input_data, bias=False),
#         #                              torch.nn.Linear(output_data, output_data, bias=False))
#         # self.inn_fc  =  lambda input_data, output_data: torch.nn.Sequential(
#         #                               torch.nn.Linear(input_data, output_data,bias=False))
#         # self.inn_fc  =  lambda input_data, output_data: torch.nn.Sequential(
#         #                              torch.nn.Linear(input_data, input_data, bias=False),
#         #                              torch.nn.Linear(input_data, output_data, bias=False),
#         #                              torch.nn.Linear(output_data, output_data, bias=False),
#         # )
#         self.dynamics =ConditionalINN(input_dim=b, condition_dim=6, hidden_dim=6, n_blocks=2)  
#         # # INITIALIZATION
#         # for module in self.modules():
#         #     if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#         #         torch.nn.init.xavier_uniform_(module.weight)


#     def forward(self, x, mode='forward'):
#         out = []
#         out_back = []
#         z = self.encoder(x.contiguous())
#         q = z.contiguous()
        
#         set_seed(2)
#         condition = torch.randn(q.shape[0],6).to('cuda')
        
#         if mode == 'forward':
#             for _ in range(self.steps):
#                 q = self.dynamics(q.squeeze(),condition)
#                 # print("Q",q)
#                 out.append(self.decoder(q.unsqueeze(1)))
#                 # print(q)

#             out.append(self.decoder(z.contiguous())) 
#             return out, out_back    

#         if mode == 'backward':
#             for _ in range(self.steps_back):
#                 q = self.dynamics(q.squeeze(),condition,reverse=True)
#                 out_back.append(self.decoder(q.unsqueeze(1)))
                
#             out_back.append(self.decoder(z.contiguous()))
#             return out, out_back

# class koopmanAE_INN(nn.Module):
#     def __init__(self, m, n, b, steps, steps_back, alpha = 1, init_scale=1):
#         super(koopmanAE_INN, self).__init__()
#         self.steps = steps
#         self.steps_back = steps_back
        
#         self.encoder = encoderNet(m, n, b, ALPHA = alpha)
#         self.decoder = decoderNet(m, n, b, ALPHA = alpha)
#         # self.inn_fc  =  lambda input_data, output_data: torch.nn.Sequential(
#         #                              torch.nn.Linear(input_data, input_data, bias=False),
#         #                              torch.nn.ReLU(),
#         #                              torch.nn.Linear(output_data, output_data, bias=False))
#         self.inn_fc  =  lambda input_data, output_data: torch.nn.Sequential(
#                                      torch.nn.Linear(input_data, input_data, bias=False),
#                                      torch.nn.Linear(output_data, output_data, bias=False))
#         # self.inn_fc  =  lambda input_data, output_data: torch.nn.Sequential(
#         #                               torch.nn.Linear(input_data, output_data,bias=False))
#         # self.inn_fc  =  lambda input_data, output_data: torch.nn.Sequential(
#         #                              torch.nn.Linear(input_data, input_data, bias=False),
#         #                              torch.nn.Linear(input_data, output_data, bias=False),
#         #                              torch.nn.Linear(output_data, output_data, bias=False),
#         # )
#         self.dynamics =inn(self.inn_fc,code_dim =b)  
#         # # INITIALIZATION
#         # for module in self.modules():
#         #     if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#         #         torch.nn.init.xavier_uniform_(module.weight)


#     def forward(self, x, mode='forward'):
#         out = []
#         out_back = []
#         z = self.encoder(x.contiguous())
#         q = z.contiguous()

        
#         if mode == 'forward':
#             for _ in range(self.steps):
#                 q,_ = self.dynamics(q,forward=True)
#                 # print("Q",q)
#                 out.append(self.decoder(q.unsqueeze(1)))

#             out.append(self.decoder(z.contiguous())) 
#             return out, out_back    

#         if mode == 'backward':
#             for _ in range(self.steps_back):
#                 q = self.dynamics(q)
#                 out_back.append(self.decoder(q.unsqueeze(1)))
                
#             out_back.append(self.decoder(z.contiguous()))
#             return out, out_back


class AffineCoupling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AffineCoupling, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.scale1 = nn.Sequential(
            nn.Linear( input_dim // 2, hidden_dim,bias=False),
            nn.Linear(hidden_dim, input_dim // 2,bias=False)
        )
        self.shift1 = nn.Sequential(
            nn.Linear( input_dim // 2, hidden_dim,bias=False),
            nn.Linear(hidden_dim, input_dim // 2,bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.Linear( input_dim // 2, hidden_dim,bias=False),
            nn.Linear(hidden_dim, input_dim // 2,bias=False),
           
        )
        self.shift2 = nn.Sequential(
            nn.Linear( input_dim // 2, hidden_dim,bias=False),
            nn.Linear(hidden_dim, input_dim // 2,bias=False),
        )

    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.input_dim // 2], x[:, self.input_dim // 2:]

        if not reverse:
            s1 = self.scale1(x2)
            s1 = torch.zeros(s1.shape).to('cuda')
            t1 = self.shift1(x2)
           
            y1 = x1 * torch.exp(s1) + t1
#             y1 = torch.mul(x1,s1) + t1

            s2 = self.scale2(y1)
            s2 = torch.zeros(s2.shape).to('cuda')
            t2 = self.shift2(y1)

           
            y2 = x2 * torch.exp(s2) + t2
#             y2 = torch.mul(x2,s2) + t2

            y = torch.cat([y1, y2], dim=1)
        else:
            s2 = self.scale2(x1)
            s2 = torch.zeros(s2.shape).to('cuda')
            t2 = self.shift2(x1)
           
            y2 = (x2 - t2) * torch.exp(-s2)
#             y2 = torch.div((x2 - t2), s2)
           
            s1 = self.scale1(y2)
            s1 = torch.zeros(s1.shape).to('cuda')
            t1 = self.shift1(y2)
           
            y1 = (x1 - t1) * torch.exp(-s1)
#             y1 = torch.div((x1 - t1),s1)
           
            y = torch.cat([y1, y2], dim=1)

        return y

class INN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_blocks):
        super(INN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks

        self.layers = nn.ModuleList([
            AffineCoupling(input_dim, hidden_dim)
            for _ in range(n_blocks)
        ])

    def forward(self, x, reverse=False):
        if not reverse:
            for layer in self.layers:
                # print(x.shape)
                x = layer(x)
        else:
            for layer in reversed(self.layers):
                x = layer(x, reverse=True)
        return x
    
class koopmanAE_INN(nn.Module):
    def __init__(self, m, n, b, steps, steps_back, alpha = 1, init_scale=1):
        super(koopmanAE_INN, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.encoder = encoderNet(m, n, b, ALPHA = alpha)
        self.decoder = decoderNet(m, n, b, ALPHA = alpha)
        # self.inn_fc  =  lambda input_data, output_data: torch.nn.Sequential(
        #                              torch.nn.Linear(input_data, input_data, bias=False),
        #                              torch.nn.ReLU(),
        #                              torch.nn.Linear(output_data, output_data, bias=False))
        # self.inn_fc  =  lambda input_data, output_data: torch.nn.Sequential(
        #                              torch.nn.Linear(input_data, input_data, bias=False),
        #                              torch.nn.Linear(output_data, output_data, bias=False))
        # self.inn_fc  =  lambda input_data, output_data: torch.nn.Sequential(
        #                               torch.nn.Linear(input_data, output_data,bias=False))
        # self.inn_fc  =  lambda input_data, output_data: torch.nn.Sequential(
        #                              torch.nn.Linear(input_data, input_data, bias=False),
        #                              torch.nn.Linear(input_data, output_data, bias=False),
        #                              torch.nn.Linear(output_data, output_data, bias=False),
        # )
        self.dynamics =INN(input_dim=b, hidden_dim=b, n_blocks=1)  
        # # INITIALIZATION
        # for module in self.modules():
        #     if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(module.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.encoder(x.contiguous())
        q = z.contiguous()
        
        if mode == 'forward':
            for _ in range(self.steps):
                # print("Q",q.shape)
                q = self.dynamics(q.squeeze(1))
                out.append(self.decoder(q.unsqueeze(1)))
                # print(q)

            out.append(self.decoder(z.contiguous())) 
            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.dynamics(q.squeeze(1),reverse=True)
                out_back.append(self.decoder(q.unsqueeze(1)))
                
            out_back.append(self.decoder(z.contiguous()))
            return out, out_back