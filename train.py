from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import socket
import sys
sys.path.insert(0, '/home/nfs/ardeshp2/torch_env/lib/python2.7/site-packages/')

from lab_imageloader import lab_imageloader
from vae import VAE

import torch
from torch.autograd import Variable

def train():

  if(len(sys.argv) == 1):
    raise NameError('[ERROR] No dataset key')
  elif(sys.argv[1] == 'lfw'):
    updates_per_epoch = 380
    log_interval = 120
    out_dir = '/data/ardeshp2/output_lfw/'
    list_dir = 'data/imglist/lfw/'
    batch_size = 32
  else:
    raise NameError('[ERROR] Incorrect key')

  data_loader = lab_imageloader(None, \
    os.path.join(out_dir, 'images'), \
    listdir=list_dir)

#  batch, batch_recon_const, _, batch_recon_const_outres = \
#    data_loader.train_next_batch(batch_size)

  model = VAE()
  print(model)
  input_color = Variable(torch.randn(16, 2, 64, 64))
  input_greylevel = Variable(torch.randn(16, 1, 64, 64))
  mu, var = model(input_color, input_greylevel)
  print(mu.size())
  print(var.size())

if __name__ == '__main__': 
  train()
