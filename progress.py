import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml
import os
import time

np.seterr(all='print')

#READ ARGS
parser = argparse.ArgumentParser()
parser.add_argument("foldername", type=str, help="path of log folder")
parser.add_argument("window_size", type=int, help="moving average window size")
args = parser.parse_args()

#SET FILE NAME
foldername = args.foldername
log_filename = os.path.join(foldername,'log.txt')
cfg_filename = os.path.join(foldername,'cfg.yaml')

#READ LOSS AND RESHAPE
attrs = []
losses = []
with open(log_filename) as file: 
    line = file.readline()
    attrs = line.split(',')[:-1]
    for line in file.readlines():
        segs = line.split(',')[1:-1]
        losses.append(segs)
losses = np.array(losses, dtype=np.float32)
losses = np.swapaxes(losses,0,1)

#MOVING AVERAGE
def moving_average_1d(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w
w_size = args.window_size
ma_losses = []
for i in range(losses.shape[0]):
    ma_losses.append(moving_average_1d(losses[i],w_size))
ma_losses = np.array(ma_losses)

#COLOR AND LINE STYLE
colors = np.random.uniform(0.0,1.0,size=(ma_losses.shape[0],3))
lw = np.ones(ma_losses.shape[0])+1
ls = ['solid']*ma_losses.shape[0]

#SET YOUR OWN COLOR AND LINE STYLE
#lw -> line width, ls -> line shape

'''colors[0] = [0.2,0.2,0.2]
lw[0] = 1
ls[0] = 'dashed'
colors[1] = [1.0,0.8,0.8]
lw[1] = 1
ls[1] = 'dashed'

colors[2] = [0.2,0.2,0.2]
lw[2] = 2
ls[2] = 'dashed'

colors[3] = [0.5,0.5,0.5]
lw[3] = 1
ls[3] = 'dashed'

colors[4] = [0.9,0.0,0.2]
lw[4] = 1

colors[5] = [1.0,0.8,0.8]
lw[5] = 1

colors[6] = [0.9,0.7,0.0]
lw[6] = 2

colors[7] = [0.0,0.4,0.8]
lw[7] = 2

colors[8] = [0.3,0.7,0.0]
lw[8] = 2

colors[9] = [0.5,0.5,0.5]
lw[9] = 1
'''



#PLT
plt.rcParams["figure.figsize"] = (6,7)
fig, axs = plt.subplots(2)
axs[0].grid(True)
axs[1].grid(True)

#UPPER GRAPH
'''lines = []
for i in range(4):
    lines.append(axs[0].plot(np.arange(ma_losses.shape[1]),ma_losses[i],color=colors[i],label=attrs[i],linewidth=lw[i],ls=ls[i])[0])
axs[0].legend(handles=lines, loc='upper right', prop={'size': 8})
axs[0].set_ylim([0,0.4])
'''

#LOWER GRAPH
lines = []
for i in range(ma_losses.shape[0]):
    lines.append(axs[1].plot(np.arange(ma_losses.shape[1]),ma_losses[i],color=colors[i],label=attrs[i],linewidth=lw[i],ls=ls[i])[0])
axs[1].legend(handles=lines, loc='upper right', prop={'size': 8})
axs[1].set_ylim([0.0,1])

#TITLE
with open(cfg_filename, "r") as stream:
    cfg = yaml.load(stream)
#title = 'FILE:%.5f  TGT%.5f' % (cfg['filename'],cfg['loss_weights']['C_TARGET'])
title = '  (%s)'%os.path.basename(os.path.normpath(foldername))
axs[1].set_title(title)

#SHOW
plt.tight_layout()
plt.show()


#plt.savefig(os.path.join(args.savepath,'%.1f.pdf' % time.time()))


        