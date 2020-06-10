import torch
import torch.nn as nn

class activation(nn.Module):
    def __init__(self):
        super(activation, self).__init__()
        self.d = 1

    def forward(self, x):
        y = (1 - (x <= 0).float()) * x 
        y = (x > 0).float() * (x < self.d).float() * (y ** 2) * 0.5 / self.d + (1 - (x > 0).float() * (x < self.d).float()) * y
        y = (x >= self.d).float() * (y - self.d / 2) + (1 - (x >= self.d).float()) * y
        return y

class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, hid_dim=16, n_layers=1):
        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.layers = [nn.Linear(self.in_dim, self.hid_dim), activation()]
        for _ in range(n_layers-1):
            self.layers.extend([nn.Linear(self.hid_dim, self.hid_dim), activation()])
        self.layers.extend([nn.Linear(self.hid_dim, self.out_dim)])

        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.sequential(x)    

class NN_g(nn.Module):
    def __init__(self, n_link=2, hid_dim=16, n_layers=1):
        super(NN_g, self).__init__()

        self.n_link = n_link
        self.control_dim = n_link
        self.in_dim = 2 * n_link + self.control_dim
        self.out_dim = 2 * n_link
        self.hid_dim = hid_dim

        self.mlp = MLP(self.in_dim, self.out_dim, hid_dim, n_layers)

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=-1)        
        return self.mlp(xu)

class NN_P(nn.Module):
    def __init__(self, n_link=2, hid_dim=16, n_layers=1):
        super(NN_P, self).__init__()

        self.n_link = n_link
        self.in_dim = 2 * n_link
        self.out_dim = 8 * (n_link**2)
        self.hid_dim = hid_dim
 
        self.mlp = MLP(self.in_dim, self.out_dim, hid_dim, n_layers)

    def forward(self, x):
        g = self.mlp(x)
        A = g[:, :self.out_dim//2].view(-1, self.in_dim, self.in_dim)
        B = g[:, self.out_dim//2:].view(-1, self.in_dim, self.in_dim)
        P = A - torch.transpose(A, 1, 2) + torch.matmul(B, torch.transpose(B, 1, 2))

        return P

class NN_pi(nn.Module):
    def __init__(self, n_link=2, hid_dim=16, n_layers=1):
        super(NN_pi, self).__init__()

        self.n_link = n_link
        self.in_dim = 2 * n_link
        self.out_dim = n_link
        self.hid_dim = hid_dim

        self.mlp = MLP(self.in_dim, self.out_dim, hid_dim, n_layers)

        self.lambd = 1e-10

    def forward(self, x):
        s = (torch.norm(x, dim=-1, keepdim=True) ** 2) / (self.lambd + (torch.norm(x, dim=-1, keepdim=True) ** 2))
        u =  s * self.mlp(x)

        return u

class NI4C(nn.Module):
    def __init__(self, n_link=2, alpha=5e-1, Q = None, Xscale=None, yscale=None):
        super(NI4C, self).__init__()

        self.n_link = n_link
        self.alpha = alpha
        if Q is None:
           self.Q = torch.eye(2 * n_link)

        self.nn_g = NN_g(n_link=n_link, hid_dim=64, n_layers=3)
        self.nn_P = NN_P(n_link=n_link, hid_dim=64, n_layers=3) 
        self.nn_pi = NN_pi(n_link=n_link, hid_dim=64, n_layers=3) 

        if Xscale is None:
            self.Xscale = torch.ones([1, 2 * n_link])
        else:
            self.Xscale = Xscale

        if yscale is None:
            self.yscale = torch.ones([1, 2 * n_link])
        else:
            self.yscale = yscale

    def forward(self, x):
        delV = torch.matmul(x, self.Q)
        V = 0.5 * torch.matmul(torch.transpose(delV.unsqueeze(-1), -1, -2), x.unsqueeze(-1)).squeeze(-1)

        P = self.nn_P(x / self.Xscale)

        PdelV = torch.matmul(P, delV.unsqueeze(-1))

        delVf = torch.matmul(torch.transpose(delV.unsqueeze(-1), -1, -2), -PdelV).squeeze(-1)
        delV_norm = torch.norm(delV, dim=-1, keepdim=True) ** 2

        dx = - PdelV.squeeze(-1) - delV * torch.relu(delVf + self.alpha * V) / (delV_norm + 1e-7)

        u = self.nn_pi(x / self.Xscale)

        effect = self.nn_g(x / self.Xscale, u)

        dx = dx - effect * self.yscale

        return dx, u, V
