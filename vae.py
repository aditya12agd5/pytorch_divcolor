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
        ngf = 128

        #Encoder Model
        self.enc_model = [nn.Conv2d(2, ngf, kernel_size=5, stride=2, padding=2),
                nn.ReLU(True),
                nn.BatchNorm2d(ngf)]
        self.enc_fc = [nn.Dropout(p=.7),
                nn.Linear(4*4*1024, self.hidden_size*2)]
                
        #Conditional Encoder model
        self.cond_model = [[nn.Conv2d(1, ngf, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf)]]

        #Build networks
        n_conv = 3
        for i in range(n_conv):
            pad_size, k_size = (1,3) if i == 2 else (2,5)
            mult = 2**i
            #Encoder model
            self.enc_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, 
                kernel_size=k_size, stride=2, padding=pad_size),
                nn.ReLU(True),
                nn.BatchNorm2d(ngf * mult * 2)]
            #Conditional model
            self.cond_model += [[nn.Conv2d(ngf * mult, ngf * mult * 2, 
                kernel_size=k_size, stride=2,padding=pad_size),
                nn.ReLU(True),
                nn.BatchNorm2d(ngf * mult * 2)]]

        #create a list of nn.Sequentials to obtain sc_feats.
        self.cond_model = [nn.Sequential(*x).cuda() for x in self.cond_model]
        self.enc_model = nn.Sequential(*self.enc_model)
        self.enc_fc = nn.Sequential(*self.enc_fc)

        #Decoder layers
        self.dec_upsamp1 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.dec_conv1 = nn.Conv2d(1024+self.hidden_size, 512, 3, stride=1, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(512)
        self.dec_upsamp2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv2 = nn.Conv2d(512*2, 256, 5, stride=1, padding=2)
        self.dec_bn2 = nn.BatchNorm2d(256)
        self.dec_upsamp3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv3 = nn.Conv2d(256*2, 128, 5, stride=1, padding=2)
        self.dec_bn3 = nn.BatchNorm2d(128)
        self.dec_upsamp4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv4 = nn.Conv2d(128*2, 64, 5, stride=1, padding=2)
        self.dec_bn4 = nn.BatchNorm2d(64)
        self.dec_upsamp5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv5 = nn.Conv2d(64, 2, 5, stride=1, padding=2)


    def decoder(self, z, sc_feat32, sc_feat16, sc_feat8, sc_feat4):
        x = z.view(-1, self.hidden_size, 1, 1)
        x = self.dec_upsamp1(x)
        x = torch.cat([x, sc_feat4], 1)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_bn1(x)
        x = self.dec_upsamp2(x) 
        x = torch.cat([x, sc_feat8], 1)
        x = F.relu(self.dec_conv2(x))
        x = self.dec_bn2(x)
        x = self.dec_upsamp3(x) 
        x = torch.cat([x, sc_feat16], 1)
        x = F.relu(self.dec_conv3(x))
        x = self.dec_bn3(x)
        x = self.dec_upsamp4(x) 
        x = torch.cat([x, sc_feat32], 1)
        x = F.relu(self.dec_conv4(x))
        x = self.dec_bn4(x)
        x = self.dec_upsamp5(x) 
        x = F.tanh(self.dec_conv5(x))
        return x
      
    #define forward pass
    def forward(self, color, greylevel, z_in, is_train=True):
        #Forward conditional 
        x = greylevel
        sc_list = []
        #Apply each sub sequential individually
        for subseq in self.cond_model:
            x = subseq(x)
            sc_list.append(x)
        sc_feat32, sc_feat16, sc_feat8, sc_feat4 = sc_list

        #Forward encoder
        x = self.enc_model(color)
        x = x.view(-1, 4*4*1024)
        x = self.enc_fc(x)
        mu = x[..., :self.hidden_size]
        logvar = x[..., self.hidden_size:]

        if(is_train == True):
            stddev = torch.sqrt(torch.exp(logvar))
            eps = Variable(torch.randn(stddev.size()).normal_()).cuda()
            z = torch.add(mu, torch.mul(eps, stddev))
        else:
            z = z_in
        color_out = self.decoder(z, sc_feat32, sc_feat16, sc_feat8, sc_feat4)

        return mu, logvar, color_out

