import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from sympyModel import nLinkPendulum
from utils import wrapToPi, numpy2torch, torch2numpy
from models import NI4C

parser = argparse.ArgumentParser(description='Paths')
parser.add_argument('--modelpath', default=None, help='Path to any pretrained model')
parser.add_argument('--savepath', default='./results', help='Path to save results')
parser.add_argument('--which', default=1, help='which link?')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dt = 0.01
n_link = 2
which = int(args.which) 
target = [np.pi, np.pi]

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

# Verification
domain = np.ones(2 * n_link)
domain[:n_link] *= np.pi
domain[n_link:] *= 10.

n_grid = 10 # number of grid points in each direction
n_samples = n_grid ** 2
theta = np.linspace(-1., 1., n_grid) * domain[which]
omega = np.linspace(-1., 1., n_grid) * domain[n_link+which]
mesh = np.meshgrid(theta, omega)
init_states = np.zeros([n_samples, 2 * n_link])
init_states[:, which] = mesh[0].reshape(-1)
init_states[:, n_link+which] = mesh[1].reshape(-1)

n_steps = 2000
ROApoints = []
grads = []
for n in range(n_samples):
    print("Sample # {}".format(n))
    x = init_states[n]
    Trj = [x.copy()]
    running_V = 100.
    inROA = True
    for t in range(n_steps):
        x_torch = numpy2torch(x.copy())
        with torch.no_grad():
            _, u, V = net(x_torch.unsqueeze(0))
            u = torch.clamp(u, -10., 10.)               # Limit the amount of control input
        u = torch2numpy(u[0])

        times = np.arange(2) * dt
        x_orig = x.copy()
        x_orig[:n_link] = wrapToPi(x_orig[:n_link] + target)
        y = odeint(system.gradient, x_orig, times, args=(u,))
        y = y[1]
        y[:n_link] = wrapToPi(y[:n_link] - target)

        if t == 0:
            grads.append(system.gradient(x_orig, times, u)) 
 
        within = (y.copy() <= domain).all()
        within = within & (y.copy() >= -domain).all()

        if not within:
            inROA = False
            break

        running_V = 0.9 * running_V + 0.1 * V.item()

        x = y
 
        Trj.append(x.copy())
 
    if (running_V > 1e-3):
        inROA = False

    if inROA:
        ROApoints.append(init_states[n])

ROApoints = np.array(ROApoints)
grads = np.array(grads)

X = init_states[:, which].reshape(n_grid, n_grid)
Y = init_states[:, which+n_link].reshape(n_grid, n_grid)
U = grads[:, which].reshape(n_grid, n_grid)
V = grads[:, which+n_link].reshape(n_grid, n_grid)

fig = plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(ROApoints[:, which], ROApoints[:, n_link+which], marker='x')
plt.axis([-domain[which], domain[which], -domain[n_link+which], domain[n_link+which]])
plt.gca().set_aspect(1./plt.gca().get_data_ratio(), 'box')
plt.xlabel('angular position (rad)', fontsize=14)
plt.ylabel('angular velocity (rad/s)', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('points $\in$ ROA', fontsize=14)

plt.subplot(1, 2, 2)
plt.streamplot(X,Y,U,V, density = 1., linewidth = 1)
plt.axis([-domain[which], domain[which], -domain[n_link+which], domain[n_link+which]])
plt.gca().set_aspect(1./plt.gca().get_data_ratio(), 'box')
plt.xlabel('angular position (rad)', fontsize=14)
plt.ylabel('angular velocity (rad/s)', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('phase potrait', fontsize=14)

plt.savefig(os.path.join(args.savepath, 'roa{}.png'.format(which)))
plt.close()
