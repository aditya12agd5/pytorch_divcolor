from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import socket
import sys
sys.path.insert(0, '/home/nfs/ardeshp2/torch_env/lib/python2.7/site-packages/')

from lab_imageloader import lab_imageloader
from vae import VAE

import torch
import torch.optim as optim
from torch.autograd import Variable

def vae_loss(mu, logvar, pred, gt, lossweights, batch_size): 
  kl_element = torch.add(torch.add(torch.add(mu.pow(2), logvar.exp()), -1), logvar.mul(-1))
  kl_loss = torch.sum(kl_element).mul(.5)
  gt = gt.view(-1, 64*64*2)
  pred = pred.view(-1, 64*64*2)
  recon_element = torch.sqrt(torch.sum(torch.mul(torch.add(gt, pred.mul(-1)).pow(2), lossweights), 1))
  recon_loss = torch.sum(recon_element).mul(1./(batch_size))
  return kl_loss.mul(1e-2)+recon_loss

def train():

  if(len(sys.argv) == 1):
    raise NameError('[ERROR] No dataset key')
  elif(sys.argv[1] == 'lfw'):
    updates_per_epoch = 380
    nepochs = 10
    log_interval = 120
    out_dir = '/data/ardeshp2/output_lfw/'
    list_dir = 'data/imglist/lfw/'
    batch_size = 32
  else:
    raise NameError('[ERROR] Incorrect key')

  data_loader = lab_imageloader(None, \
    os.path.join(out_dir, 'images'), \
    listdir=list_dir)

  model = VAE()
  model.cuda()
  print(model)

  optimizer = optim.Adam(model.parameters(), lr=1e-5)

  for epochs in range(nepochs):
    train_loss = 0.
    for i in range(updates_per_epoch):
      batch, batch_recon_const, batch_lossweights, batch_recon_const_outres = \
          data_loader.train_next_batch(batch_size)

      input_color = Variable(torch.from_numpy(batch)).cuda()
      lossweights = Variable(torch.from_numpy(batch_lossweights)).cuda()
      input_greylevel = Variable(torch.from_numpy(batch_recon_const)).cuda()
      
      optimizer.zero_grad()
      mu, logvar, color_out = model(input_color, input_greylevel)
      loss = vae_loss(mu, logvar, color_out, input_color, lossweights, batch_size)
      loss.backward()
      optimizer.step()

      train_loss = train_loss + loss.data[0]

      if(i % log_interval == 0):
        data_loader.save_output_with_gt(color_out.cpu().data.numpy(), \
          batch, \
          'train_%05d_%05d' % (epochs, i), \
          batch_size, \
          net_recon_const=batch_recon_const_outres) 

    print('[DEBUG] Epoch %d has loss %f' % (epochs, train_loss)) 

if __name__ == '__main__': 
  train()
