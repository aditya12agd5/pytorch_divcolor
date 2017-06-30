import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class MDN(nn.Module):
  
  #define layers
  def __init__(self):
    super(MDN, self).__init__()

    self.feats_nch = 512
    self.hidden_size = 64
    self.nmix = 8
    self.nout = (self.hidden_size+1)*self.nmix

    #MDN Layers
    self.mdn_conv1 = nn.Conv2d(self.feats_nch, 384, 5, stride=1, padding=2)
    self.mdn_bn1 = nn.BatchNorm2d(384)
    self.mdn_conv2 = nn.Conv2d(384, 320, 5, stride=1, padding=2)
    self.mdn_bn2 = nn.BatchNorm2d(320)
    self.mdn_conv3 = nn.Conv2d(320, 288, 5, stride=1, padding=2)
    self.mdn_bn3 = nn.BatchNorm2d(288)
    self.mdn_conv4 = nn.Conv2d(288, 256, 5, stride=2, padding=2)
    self.mdn_bn4 = nn.BatchNorm2d(256)
    self.mdn_conv5 = nn.Conv2d(256, 128, 5, stride=1, padding=2)
    self.mdn_bn5 = nn.BatchNorm2d(128)
    self.mdn_conv6 = nn.Conv2d(128, 96, 5, stride=2, padding=2)
    self.mdn_bn6 = nn.BatchNorm2d(96)
    self.mdn_conv7 = nn.Conv2d(96, 64, 5, stride=2, padding=2)
    self.mdn_bn7 = nn.BatchNorm2d(64)
    self.mdn_dropout1 = nn.Dropout(p=.7)
    self.mdn_fc1 = nn.Linear(4*4*64, self.nout)

  #define forward pass
  def forward(self, feats):
    x = F.relu(self.mdn_conv1(feats))
    x = self.mdn_bn1(x)
    x = F.relu(self.mdn_conv2(x))
    x = self.mdn_bn2(x)
    x = F.relu(self.mdn_conv3(x))
    x = self.mdn_bn3(x)
    x = F.relu(self.mdn_conv4(x))
    x = self.mdn_bn4(x)
    x = F.relu(self.mdn_conv5(x))
    x = self.mdn_bn5(x)
    x = F.relu(self.mdn_conv6(x))
    x = self.mdn_bn6(x)
    x = F.relu(self.mdn_conv7(x))
    x = self.mdn_bn7(x)
    x = x.view(-1, 4*4*64)
    x = self.mdn_dropout1(x)
    x = self.mdn_fc1(x)
    return x
