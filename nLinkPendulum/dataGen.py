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

def generate(n_link=2, n_seqs=100, seq_len=500, dt=0.01, seed=0):
    system = nLinkPendulum(n=n_link)

    states = np.zeros((n_seqs, seq_len, 2 * n_link))
    grad_states = np.zeros_like(states)
    grad_states_forced = np.zeros_like(states)
    forces = np.zeros((n_seqs, seq_len, n_link)) 
    for n in range(n_seqs):
        np.random.seed(seed+n)
        init_pos = 2 * np.pi * np.random.rand(n_link) - np.pi
        init_vel = 0
        init_state = np.concatenate([np.broadcast_to(init_pos, n_link),
                                     np.broadcast_to(init_vel, n_link)])
        times = np.arange(2) * dt      
        
        for t in range(seq_len):
            if t == 0:
                y = init_state
            else:
                y = np.concatenate([wrapToPi(state[1, :n_link]), state[1, n_link:]])    
            state = odeint(system.gradient, y.copy(), times)
            grad_state = system.gradient(y.copy(), times) 
            f = 100 * np.random.rand(n_link) - 50.
            grad_state_forced = system.gradient(y.copy(), times, u=f)
            states[n, t, :n_link] = wrapToPi(state[0, :n_link])
            states[n, t, n_link:] = state[0, n_link:]
            grad_states[n, t] = grad_state
            grad_states_forced[n, t] = grad_state_forced
            forces[n, t] = f

    return states, grad_states, grad_states_forced, forces

if args.set == 'train':
    seed = 0
    n_seqs = 400
elif args.set == 'val':
    seed = 500
    n_seqs = 50
else:
    seed = 1000
    n_seqs = 1

savepath = args.savepath + '/' + args.set

states, grads, grads_forced, forces = generate(n_link=int(args.nlink), n_seqs=int(n_seqs), seq_len=500, seed=int(seed), dt=0.01)
dictionary = {}
dictionary['states'] = states
dictionary['grads'] = grads
dictionary['grads_forced'] = grads_forced 
dictionary['forces'] = forces
np.save(savepath, dictionary)

animate_pendulum(states[np.random.randint(int(n_seqs))][::2], dt=0.01, savepath=args.savepath+'/demo')
