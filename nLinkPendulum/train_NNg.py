import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils import wrapToPi, numpy2torch
from models import NN_g

parser = argparse.ArgumentParser(description='Paths')
parser.add_argument('--dataset', default='./data', help='dataset path')
parser.add_argument('--savepath', default='./saved_models', help='Save path')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

n_link = 2
dt = 0.01

# Target
target = [np.pi, np.pi] 
target = np.array(target)[np.newaxis, :] 

# Training data
train_data = np.load(os.path.join(args.dataset, 'train.npy'), allow_pickle=True).item()

train_X = train_data['states']
train_X = train_X.reshape(-1, 2 * n_link) 
train_X[:, :n_link] = wrapToPi(train_X[:, :n_link] - target)

train_y = train_data['grads']
train_y = train_y.reshape(-1, 2 * n_link) 

train_y_forced = train_data['grads_forced']
train_y_forced = train_y_forced.reshape(-1, 2 * n_link)

train_u = train_data['forces']
train_u = train_u.reshape(-1, n_link)

Xscale = np.max(np.abs(train_X), axis=0, keepdims=True)
yscale = np.max(np.abs(train_y), axis=0, keepdims=True)
yfscale = np.max(np.abs(train_y_forced), axis=0, keepdims=True)
yscale = np.maximum(yscale, yfscale)

np.save(args.savepath + '/Xscale.npy', np.array(Xscale))
np.save(args.savepath + '/yscale.npy', np.array(yscale))

train_X /= Xscale
train_y /= yscale
train_y_forced /= yscale

order = np.arange(len(train_X))
np.random.shuffle(order)
train_X = train_X[order]
train_y = train_y[order]
train_y_forced = train_y_forced[order]
train_u = train_u[order]

train_X = numpy2torch(train_X)
train_y = numpy2torch(train_y)
train_y_forced = numpy2torch(train_y_forced)
train_u = numpy2torch(train_u)

# Validation data
val_data = np.load(os.path.join(args.dataset, 'val.npy'), allow_pickle=True).item()

val_X = val_data['states']
val_X = val_X.reshape(-1, 2 * n_link)
val_X[:, :n_link] = wrapToPi(val_X[:, :n_link] - target)

val_y = val_data['grads']
val_y = val_y.reshape(-1, 2 * n_link)

val_y_forced = val_data['grads_forced']
val_y_forced = val_y_forced.reshape(-1, 2 * n_link)

val_u = val_data['forces']
val_u = val_u.reshape(-1, n_link)

val_X /= Xscale
val_y /= yscale
val_y_forced /= yscale

val_X = numpy2torch(val_X)
val_y = numpy2torch(val_y)
val_y_forced = numpy2torch(val_y_forced)
val_u = numpy2torch(val_u)

# Define network
net = NN_g(n_link=n_link, hid_dim=64, n_layers=3)
print(net)

# Define optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=3e-7)

# Define loss
criterion = nn.MSELoss()

# Training and validation
writer = SummaryWriter()

print(len(train_X))

num_epochs = 300
batch_size = 32
train_steps = len(train_X)//batch_size
val_steps = len(val_X)//batch_size
print("Starting training\n")
for epoch in range(num_epochs):
    train_error = 0.
    for step in range(train_steps):
        X = train_X[step*batch_size : (step+1)*batch_size]
        y = train_y[step*batch_size : (step+1)*batch_size]
        y_forced = train_y_forced[step*batch_size : (step+1)*batch_size]
        u = train_u[step*batch_size : (step+1)*batch_size]

        diff = net(X, u) 
        y_forced_pred = y + diff

        train_loss = criterion(y_forced_pred, y_forced)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_error += train_loss.item()

    with torch.no_grad():
        val_error = 0.
        for vstep in range(val_steps):
            X = val_X[vstep*batch_size : (vstep+1)*batch_size]
            y = val_y[vstep*batch_size : (vstep+1)*batch_size]
            y_forced = val_y_forced[vstep*batch_size : (vstep+1)*batch_size]
            u = val_u[vstep*batch_size : (vstep+1)*batch_size]
 
            diff = net(X, u)
            y_forced_pred = y + diff

            val_loss = criterion(y_forced_pred, y_forced)
            val_error += val_loss.item()

    print("Epoch: {}, Training Error: {}, Validation Error: {}".format(epoch,
                                                                       train_error/train_steps,
                                                                       val_error/val_steps))

    writer.add_scalar('Loss/train/NNg', train_error/train_steps, epoch)
    writer.add_scalar('Loss/val/NNg', val_error/val_steps, epoch)

    torch.save(net.state_dict(), args.savepath + '/NN_g-{0:04d}.pt'.format(epoch))

    for g in optimizer.param_groups:
        if g['lr'] > 0.0001:
            g['lr'] *= 0.99
        print("LR: {}".format(g['lr']))
