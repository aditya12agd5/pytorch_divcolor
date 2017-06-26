from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import socket
import sys
sys.path.insert(0, '/home/nfs/ardeshp2/torch_env/lib/python2.7/site-packages/')
import numpy as np 

from lab_imageloader import lab_imageloader
from vae import VAE
from mdn import MDN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

batch_size = 32
hidden_size = 64
nmix = 8

def get_params(): 
  if(len(sys.argv) == 1):
    raise NameError('[ERROR] No dataset key')
  elif(sys.argv[1] == 'lfw'):
    updates_per_epoch = 380
    nepochs = 10
    log_interval = 120
    out_dir = 'data/output/lfw/'
    listdir = 'data/imglist/lfw/'
    featslistdir = 'data/featslist/lfw/'
  else:
    raise NameError('[ERROR] Incorrect key')
  return updates_per_epoch, nepochs, log_interval, out_dir, listdir, featslistdir

def vae_loss(mu, logvar, pred, gt, lossweights, batch_size): 
  kl_element = torch.add(torch.add(torch.add(mu.pow(2), logvar.exp()), -1), logvar.mul(-1))
  kl_loss = torch.sum(kl_element).mul(.5)
  gt = gt.view(-1, 64*64*2)
  pred = pred.view(-1, 64*64*2)
  recon_element = torch.sqrt(torch.sum(torch.mul(torch.add(gt, pred.mul(-1)).pow(2), lossweights), 1))
  recon_loss = torch.sum(recon_element).mul(1./(batch_size))
  return kl_loss.mul(1e-2)+recon_loss

def get_gmm_coeffs(gmm_params):
  gmm_mu = gmm_params[..., :hidden_size*nmix]
  gmm_mu.contiguous()
  gmm_pi_activ = gmm_params[..., hidden_size*nmix:]
  gmm_pi_activ.contiguous()
  gmm_pi = F.softmax(gmm_pi_activ)
  return gmm_mu, gmm_pi

def mdn_loss(gmm_params, mu, stddev, batch_size):
  gmm_mu, gmm_pi = get_gmm_coeffs(gmm_params)
  eps = Variable(torch.randn(stddev.size()).normal_()).cuda() 
  z = torch.add(mu, torch.mul(eps, stddev))
  z_flat = z.repeat(1, nmix)
  z_flat = z_flat.view(batch_size*nmix, hidden_size)
  gmm_mu_flat = gmm_mu.view(batch_size*nmix, hidden_size)
  dist_all = torch.sqrt(torch.sum(torch.add(z_flat, gmm_mu_flat.mul(-1)).pow(2).mul(100), 1))
  dist_all = dist_all.view(batch_size, nmix)
  dist_min, selectids = torch.min(dist_all, 1)
  gmm_pi_min = torch.gather(gmm_pi, 1, selectids)
  gmm_loss = torch.sum(torch.add(torch.log(gmm_pi_min+1e-30), dist_min)).mul(1./batch_size)
  return gmm_loss

def train_vae():
  updates_per_epoch, nepochs, log_interval, out_dir, listdir, featslistdir = \
      get_params()

  data_loader = lab_imageloader(\
    os.path.join(out_dir, 'images'), \
    listdir=listdir,\
    featslistdir=featslistdir)

  model = VAE()
  model.cuda()

  optimizer = optim.Adam(model.parameters(), lr=1e-5)

  for epochs in range(nepochs):
    train_loss = 0.
    data_loader.random_reset()
    for i in range(updates_per_epoch):
      batch, batch_recon_const, batch_lossweights, batch_recon_const_outres, _ = \
          data_loader.train_next_batch(batch_size)

      input_color = Variable(torch.from_numpy(batch)).cuda()
      lossweights = Variable(torch.from_numpy(batch_lossweights)).cuda()
      input_greylevel = Variable(torch.from_numpy(batch_recon_const)).cuda()
      z = Variable(torch.randn(batch_size, hidden_size))
 
      optimizer.zero_grad()
      mu, logvar, color_out = model(input_color, input_greylevel, z)
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
    train_loss = (train_loss*1.)/(updates_per_epoch)
    print('[DEBUG] Training VAE, epoch %d has loss %f' % (epochs, train_loss)) 
    torch.save(model.state_dict(), '%s/models/model_vae_%03d' % (out_dir, epochs))

def train_mdn(nepochs_mdn=5):
  updates_per_epoch, nepochs, log_interval, out_dir, listdir, featslistdir = \
      get_params()

  data_loader = lab_imageloader(\
    os.path.join(out_dir, 'images'), \
    listdir=listdir,\
    featslistdir=featslistdir)
  data_loader.random_reset()

  model_vae = VAE()
  model_vae.cuda()
  model_vae.load_state_dict(torch.load('%s/models/model_vae_%03d' % (out_dir, nepochs-1)))

  model_mdn = MDN()
  model_mdn.cuda()

  optimizer = optim.Adam(model_mdn.parameters(), lr=1e-5)

  n_train_batches = np.int_(np.floor((data_loader.train_img_num*1.)/batch_size))
  for epochs_mdn in range(nepochs_mdn):
    train_loss = 0.
    for i in range(n_train_batches):
      batch, batch_recon_const, batch_lossweights, batch_recon_const_outres, batch_feats = \
        data_loader.train_next_batch(batch_size, get_feats=True)
      input_color = Variable(torch.from_numpy(batch)).cuda()
      input_greylevel = Variable(torch.from_numpy(batch_recon_const)).cuda()
      input_feats = Variable(torch.from_numpy(batch_feats)).cuda()
      z = Variable(torch.randn(batch_size, hidden_size))

      optimizer.zero_grad()
      mu, logvar, _ = model_vae(input_color, input_greylevel, z)

      mdn_gmm_params = model_mdn(input_feats)

      loss = mdn_loss(mdn_gmm_params, mu, torch.sqrt(torch.exp(logvar)), batch_size)
      loss.backward()
      optimizer.step()
      train_loss = train_loss + loss.data[0]
    train_loss = (train_loss*1.)/(n_train_batches)
    print('[DEBUG] Training MDN, epoch %d has loss %f' % (epochs_mdn, train_loss))
    torch.save(model_mdn.state_dict(), '%s/models_mdn/model_mdn_%03d' % (out_dir, epochs_mdn))

def divcolor(num_batches=4, nepochs_mdn=5):
  _, nepochs, _, out_dir, listdir, featslistdir = \
      get_params()

  data_loader = lab_imageloader(\
    os.path.join(out_dir, 'images'), \
    listdir=listdir,\
    featslistdir=featslistdir)
  data_loader.reset()

  model_vae = VAE()
  model_vae.cuda()
  model_vae.load_state_dict(torch.load('%s/models/model_vae_%03d' % (out_dir, nepochs-1)))

  model_mdn = MDN()
  model_mdn.cuda()
  model_mdn.load_state_dict(torch.load('%s/models_mdn/model_mdn_%03d' % (out_dir, nepochs_mdn-1)))

  for i in range(num_batches):
    batch, batch_recon_const, batch_recon_const_outres, batch_names, batch_feats = \
      data_loader.test_next_batch(batch_size, get_feats=True)

    input_feats = Variable(torch.from_numpy(batch_feats)).cuda()

    mdn_gmm_params = model_mdn(input_feats)
    gmm_mu, gmm_pi = get_gmm_coeffs(mdn_gmm_params)
    gmm_pi = gmm_pi.view(-1, 1)
    gmm_mu = gmm_mu.view(-1, hidden_size)

    for j in range(batch_size):
      batch_j = np.tile(batch[j, ...], (batch_size, 1, 1, 1))
      batch_recon_const_j = np.tile(batch_recon_const[j, ...], (batch_size, 1, 1, 1))
      batch_recon_const_outres_j = np.tile(batch_recon_const_outres[j, ...], (batch_size, 1, 1, 1))

      input_color = Variable(torch.from_numpy(batch_j)).cuda()
      input_greylevel = Variable(torch.from_numpy(batch_recon_const_j)).cuda()
 
      curr_mu = gmm_mu[j*nmix:(j+1)*nmix, :]
      orderid = np.argsort(\
        gmm_pi[j*nmix:(j+1)*nmix, 0].cpu().data.numpy().reshape(-1))
      
      z = curr_mu.repeat(np.int_((batch_size*1.)/nmix), 1)

      _, _, color_out = model_vae(input_color, input_greylevel, z, is_train=False)

      data_loader.save_output_with_gt(color_out.cpu().data.numpy()[orderid, ...], \
       batch_j[orderid, ...], \
       'divcolor_%05d_%05d' % (i, j), \
       nmix, \
       net_recon_const=batch_recon_const_outres_j[orderid, ...]) 

if __name__ == '__main__': 
  train_vae()
  train_mdn()
  divcolor()
