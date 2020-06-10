import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils import wrapToPi, numpy2torch
from models import NI4C

parser = argparse.ArgumentParser(description='Paths')
parser.add_argument('--dataset', default='./data', help='dataset path')
parser.add_argument('--savepath', default='./saved_models', help='Save path')
parser.add_argument('--NNg', default='./saved_models/NN_g.pt', help='path to trained NN_g')
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

Xscale = np.load(os.path.join(os.path.dirname(args.NNg), 'Xscale.npy'))
yscale = np.load(os.path.join(os.path.dirname(args.NNg), 'yscale.npy'))
train_y /= yscale

order = np.arange(len(train_X))
np.random.shuffle(order)
train_X = train_X[order]
train_y = train_y[order]

train_X = numpy2torch(train_X)
train_y = numpy2torch(train_y)

# Validation data
val_data = np.load(os.path.join(args.dataset, 'val.npy'), allow_pickle=True).item()

val_X = val_data['states']
val_X = val_X.reshape(-1, 2 * n_link)
val_X[:, :n_link] = wrapToPi(val_X[:, :n_link] - target)

val_y = val_data['grads']
val_y = val_y.reshape(-1, 2 * n_link)

val_y /= yscale

val_X = numpy2torch(val_X)
val_y = numpy2torch(val_y)

Xscale = numpy2torch(Xscale)
yscale = numpy2torch(yscale)

# Define network
net = NI4C(n_link=n_link, Xscale=Xscale, yscale=yscale)
print(net)
checkpoint = torch.load(args.NNg)
checkpoint = {'nn_g.' + k : v for k, v in checkpoint.items()}
net.load_state_dict(checkpoint, strict=False)

# Define optimizer
for name, param in net.named_parameters():
    if name[:4] == 'nn_g':
        param.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)

# Define loss
criterion = nn.MSELoss()

# Training and validation
writer = SummaryWriter()

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
        
        y_pred, _, _ = net(X)
        y_pred = y_pred / yscale
  
        train_loss = criterion(y_pred, y)  
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_error += train_loss.item()

    with torch.no_grad():
        val_error = 0.
        for vstep in range(val_steps):
            X = val_X[vstep*batch_size : (vstep+1)*batch_size]
            y = val_y[vstep*batch_size : (vstep+1)*batch_size]

            y_pred, _, _ = net(X)
            y_pred = y_pred / yscale

            val_loss = criterion(y_pred, y)
            val_error += val_loss.item()

    print("Epoch: {}, Training Error: {}, Validation Error: {}".format(epoch,
                                                                       train_error/train_steps,
                                                                       val_error/val_steps))

    writer.add_scalar('Loss/train/NNpiP', train_error/train_steps, epoch)
    writer.add_scalar('Loss/val/NNpiP', val_error/val_steps, epoch)

    torch.save(net.state_dict(), os.path.join(args.savepath, 'NI4C-{0:04d}.pt'.format(epoch)))

    for g in optimizer.param_groups:
        if g['lr'] > 0.0001:
            g['lr'] *= 0.99
        print("LR: {}".format(g['lr']))
