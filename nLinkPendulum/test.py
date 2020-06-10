import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from sympyModel import nLinkPendulum
from utils import wrapToPi, animate_pendulum, numpy2torch, torch2numpy
from models import NI4C

parser = argparse.ArgumentParser(description='Paths')
parser.add_argument('--modelpath', default=None, help='Path to any pretrained model')
parser.add_argument('--savepath', default='./results', help='Path to save result')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dt = 0.01
n_link = 2

# Initial state
np.random.seed(0)
init_state = 2 * np.random.rand(2 * n_link) - 1
init_state[:n_link] *= np.pi

# Target 
target = [np.pi, np.pi]
target = np.array(target)

# Load scaling
Xscale = np.load(os.path.join(os.path.dirname(args.modelpath), 'Xscale.npy'))
yscale = np.load(os.path.join(os.path.dirname(args.modelpath), 'yscale.npy'))

Xscale = numpy2torch(Xscale)
yscale = numpy2torch(yscale)

# Load network
alpha = 5e-1
Q = torch.diagflat(torch.Tensor([0.60, 0.32, 0.045, 0.035]))
net = NI4C(n_link=n_link, alpha=alpha, Q=Q, yscale=yscale, Xscale=Xscale)
checkpoint = torch.load(args.modelpath)
net.load_state_dict(checkpoint)

# Load SymPy model
system = nLinkPendulum(n=n_link)

# Apply control and simulate
Trj = [init_state.copy()]
Vs = []
t = 0
x = init_state
while t < 2000:
     print(t)

     x_torch = numpy2torch(x.copy())
     with torch.no_grad():
         _, u, V = net(x_torch.unsqueeze(0))
         u = torch.clamp(u, -10, 10)               # Limit the amount of control input
     u = torch2numpy(u[0])

     times = np.arange(2) * dt
     x_orig = x.copy()
     x_orig[:n_link] = wrapToPi(x_orig[:n_link] + target)
     y = odeint(system.gradient, x_orig, times, args=(u,))
     y = y[1]
     y[:n_link] = wrapToPi(y[:n_link] - target)

     x = y

     Trj.append(x.copy())

     Vs.append(V.item())

     t += 1
 
Trj = np.array(Trj[:-1])

# Plot response
plt.plot(Trj)
plt.title('Closed-loop response', fontsize=14)
plt.savefig(os.path.join(args.savepath, 'response.png'))
plt.show()

# Plot Lyapunov energy
plt.plot(Vs)
plt.title('Lyapunov energy', fontsize=14)
plt.savefig(os.path.join(args.savepath, 'energy.png'))
plt.show()

# Make video
print('Saving video ...')
Trj[:, :n_link] = wrapToPi(Trj[:, :n_link] + target)
animate_pendulum(Trj[::2], dt, target, savepath=os.path.join(args.savepath, 'demo'))
