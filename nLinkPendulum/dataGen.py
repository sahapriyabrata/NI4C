import argparse
import numpy as np
from scipy.integrate import odeint

from sympyModel import nLinkPendulum
from utils import wrapToPi, animate_pendulum

parser = argparse.ArgumentParser(description='Args')
parser.add_argument('--nlink', default=2, help='number of links')
parser.add_argument('--set', default='test', help='train, val or test')
parser.add_argument('--savepath', default='./data', help='Save path')
args = parser.parse_args()

def generate(n_link=2, n_samples=10000, dt=0.01, seed=0):
    np.random.seed(seed)
    system = nLinkPendulum(n=n_link)

    states = np.zeros((n_samples, 2 * n_link))
    states[:, :n_link] = 2 * np.pi * np.random.rand(n_samples, n_link) - np.pi 
    states[:, :n_link] = states[:, :n_link] / 3 + np.pi
    states[:, n_link:] = (20/3) * np.random.rand(n_samples, n_link) - 10/3

    grad_states = np.zeros_like(states)
    grad_states_forced = np.zeros_like(states)
    forces = np.zeros((n_samples, n_link))

    times = np.arange(2) * dt
    for n in range(n_samples):
        grad_states[n] = system.gradient(states[n], times)
        f = 20 * np.random.rand(n_link) - 10.
        grad_states_forced[n] = system.gradient(states[n], times, u=f)
        forces[n] = f

    return states, grad_states, grad_states_forced, forces

if args.set == 'train':
    seed = 0
    n_samples = 10000
elif args.set == 'val':
    seed = 500
    n_samples = 5000
else:
    seed = 1000
    n_seqs = 1

savepath = args.savepath + '/' + args.set

states, grads, grads_forced, forces = generate(n_link=int(args.nlink), n_samples=int(n_samples), dt=0.01, seed=int(seed))
dictionary = {}
dictionary['states'] = states
dictionary['grads'] = grads
dictionary['grads_forced'] = grads_forced 
dictionary['forces'] = forces
np.save(savepath, dictionary)
