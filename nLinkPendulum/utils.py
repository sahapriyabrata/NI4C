import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
import math

def numpy2torch(q):
    q = torch.from_numpy(q).float()
    if torch.cuda.is_available():
        q = q.cuda()
    return q

def torch2numpy(q):
    if q.is_cuda:
        q = q.cpu()
    return q.numpy()

def wrapToPi(x):
    xwrap = np.remainder(x, 2*np.pi)
    idx = np.abs(xwrap) > np.pi
    xwrap[idx] -= 2 * np.pi * np.sign(xwrap[idx])
    return xwrap

def torch_wrapToPi(x):
    x = x.float()
    xwrap = torch.fmod(x, 2 * math.pi)
    idx = torch.abs(xwrap) > math.pi
    xwrap[idx] -= 2 * math.pi * torch.sign(xwrap[idx])
    return xwrap

def get_xy_coords(p, lengths=None):
    """Get (x, y) coordinates from generalized coordinates p"""
    p = np.atleast_2d(p)
    n = p.shape[1] // 2
    if lengths is None:
        lengths = np.ones(n) / n
    zeros = np.zeros(p.shape[0])[:, None]
    x = np.hstack([zeros, lengths * np.sin(p[:, :n])])
    y = np.hstack([zeros, -lengths * np.cos(p[:, :n])])
    return np.cumsum(x, 1), np.cumsum(y, 1)

def animate_pendulum(p, dt, target=None, interval=1000, fps=30, dpi=500, savepath='./demo.mp4'):
    t = np.arange(len(p)) * dt
    x, y = get_xy_coords(p)

    if target is not None:
        target = np.pad(target, (0, len(target)), 'constant')
        x_tar, y_tar = get_xy_coords(target[np.newaxis])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))

    if target is not None:
        ax.plot(x_tar[0], y_tar[0], 'ro--', lw=2)

    # display the current time
    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)

    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        time_text.set_text('')
        line.set_data([], [])
        return line,

    def animate(i):
        time_text.set_text('time = {:2.2f}'.format(t[i]))
        line.set_data(x[i], y[i])
        return line,

    anim = animation.FuncAnimation(fig, animate, frames=len(t),
                                   interval=interval * t.max() / len(t),
                                   blit=True, init_func=init)

    videowriter = animation.FFMpegWriter(fps=fps)
    anim.save(savepath + '.mp4', writer=videowriter, dpi=dpi)
    plt.close(fig)
