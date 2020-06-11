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
parser.add_argument('--init', default='grid', help='initial points? grid : random')
parser.add_argument('--which', default=1, help='required only for grid initial points. which link?')
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

if args.init == 'grid':
    n_grid = 10
    n_samples = n_grid ** 2
    theta = np.linspace(-1., 1., n_grid) * domain[which]
    omega = np.linspace(-1., 1., n_grid) * domain[n_link+which]
    mesh = np.meshgrid(theta, omega)
    init_states = np.zeros([n_samples, 2 * n_link])
    init_states[:, which] = mesh[0].reshape(-1)
    init_states[:, n_link+which] = mesh[1].reshape(-1)
else: # random samples
    n_samples = 100
    init_states = 2 * np.random.rand(n_samples, 2 * n_link) - 1
    init_states *= domain[np.newaxis, :]

n_steps = 2000
Trjs = []
for n in range(n_samples):
    print("Sample # {}".format(n))
    x = init_states[n]
    Trj = [x.copy()]
    running_V = 100.
    valid = True
    for t in range(n_steps):
        x_torch = numpy2torch(x.copy())
        with torch.no_grad():
            _, u, V = net(x_torch.unsqueeze(0))
            u = torch.clamp(u, -7.5, 7.5)               # Limit the amount of control input
        u = torch2numpy(u[0])

        times = np.arange(2) * dt
        x_orig = x.copy()
        x_orig[:n_link] = wrapToPi(x_orig[:n_link] + target)
        y = odeint(system.gradient, x_orig, times, args=(u,))
        y = y[1]
        y[:n_link] = wrapToPi(y[:n_link] - target)
 
        within = (y.copy() <= domain).all()
        within = within & (y.copy() >= -domain).all()

        if not within:
            valid = False
            break

        running_V = 0.9 * running_V + 0.1 * V.item()

        x = y
 
        Trj.append(x.copy())
    
    if (running_V > 1e-2):
        valid = False

    if valid:
        Trjs.append(Trj)

Trjs = np.array(Trjs)

if args.init == 'grid':
    plt.scatter(Trjs[:, 0, which], Trjs[:, 0, n_link+which], marker='x')
    plt.axis([-np.pi, np.pi, -10., 10.])
    plt.gca().set_aspect(1./plt.gca().get_data_ratio(), 'box')
    plt.xlabel('angular position (rad)', fontsize=14)
    plt.ylabel('angular velocity (rad/s)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(os.path.join(args.savepath, 'roa{}.png'.format(which)))
    plt.close()
else:  
    for i in range(n_link):
        plt.scatter(Trjs[:, 0, i], Trjs[:, 0, n_link+i], marker='x')
        plt.axis([-np.pi, np.pi, -10., 10.])
        plt.gca().set_aspect(1./plt.gca().get_data_ratio(), 'box')
        plt.xlabel('angular position (rad)', fontsize=14)
        plt.ylabel('angular velocity (rad/s)', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(os.path.join(args.savepath, 'roa{}.png'.format(i)))
        plt.close()
