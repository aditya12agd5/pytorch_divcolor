import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class VAE(nn.Module):
  
  #define layers
  def __init__(self):
    super(VAE, self).__init__()
    self.hidden_size = 64
    self.enc_conv1 = nn.Conv2d(2, 128, 5, stride=2, padding=2)
    self.enc_bn1 = nn.BatchNorm2d(128)
    self.enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
    self.enc_bn2 = nn.BatchNorm2d(256)
    self.enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
    self.enc_bn3 = nn.BatchNorm2d(512)
    self.enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
    self.enc_bn4 = nn.BatchNorm2d(1024)
    self.enc_fc1 = nn.Linear(4*4*1024, self.hidden_size*2)
    self.enc_dropout1 = nn.Dropout(p=.7)

    self.cond_enc_conv1 = nn.Conv2d(1, 128, 5, stride=2, padding=2)
    self.cond_enc_bn1 = nn.BatchNorm2d(128)
    self.cond_enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
    self.cond_enc_bn2 = nn.BatchNorm2d(256)
    self.cond_enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
    self.cond_enc_bn3 = nn.BatchNorm2d(512)
    self.cond_enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
    self.cond_enc_bn4 = nn.BatchNorm2d(1024)

  def encoder(self, x):
    x = F.relu(self.enc_conv1(x))
    x = self.enc_bn1(x)
    x = F.relu(self.enc_conv2(x))
    x = self.enc_bn2(x)
    x = F.relu(self.enc_conv3(x))
    x = self.enc_bn3(x)
    x = F.relu(self.enc_conv4(x))
    x = self.enc_bn4(x)
    x = x.view(-1, 4*4*1024)
    x = self.enc_dropout1(x)
    x = self.enc_fc1(x)
    mu = x[..., :self.hidden_size]
    var = x[..., self.hidden_size:]
    return mu, var

  def cond_encoder(self, x):
    x = F.relu(self.cond_enc_conv1(x))
    sc_feat32 = self.cond_enc_bn1(x)
    x = F.relu(self.cond_enc_conv2(sc_feat32))
    sc_feat16 = self.cond_enc_bn2(x)
    x = F.relu(self.cond_enc_conv3(sc_feat16))
    sc_feat8 = self.cond_enc_bn3(x)
    x = F.relu(self.cond_enc_conv4(sc_feat8))
    sc_feat4 = self.cond_enc_bn4(x)
    return sc_feat32, sc_feat16, sc_feat8, sc_feat4

#  def decoder(self, z, sc_feat32, sc_feat16, sc_feat8, sc_feat4):

  #define forward pass
  def forward(self, color, greylevel):
    mu, var = self.encoder(color)
    sc_feat32, sc_feat16, sc_feat8, sc_feat4 = self.cond_encoder(greylevel)
    return mu, var
