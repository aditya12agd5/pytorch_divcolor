import torch
import visdom
import numpy as np

class Logger():

  def __init__(self, server, port, outdir):
    self.vis = visdom.Visdom(port=port, server=server)

    titles = ['VAE -- KL Div', 'VAE -- Weighted L2', 'VAE -- L2']
    self.vis_plot_vae = []
    for title in titles:
      self.vis_plot_vae.append(self.vis.line(
        X=np.array([0.], dtype='f'),
        Y=np.array([0.], dtype='f'),
        opts=dict(
          xlabel='Iteration',\
          ylabel='Loss',\
          title=title)))

    self.vis_plot_test_vae = self.vis.line(
        X=np.array([0.], dtype='f'),
        Y=np.array([0.], dtype='f'),
        opts=dict(
          xlabel='Iteration',\
          ylabel='Test Loss',\
          title='VAE Test Loss'))


    self.vis_plot_mdn = []
    titles = ['MDN Loss', 'MDN -- L2']
    for title in titles:
      self.vis_plot_mdn.append(self.vis.line(
        X=np.array([0.], dtype='f'),
        Y=np.array([0.], dtype='f'),
        opts=dict(
          xlabel='Iteration',\
          ylabel='Loss',\
          title=title)))

    self.fp_vae = open('%s/log_vae.txt' % outdir, 'w')
    self.fp_vae.write('Iteration; KLDiv; WeightedL2; L2;\n')
    self.fp_vae.flush()

    self.fp_test_vae = open('%s/log_test_vae.txt' % outdir, 'w')
    self.fp_test_vae.write('Iteration; Loss;\n')
    self.fp_test_vae.flush()

    self.fp_mdn = open('%s/log_mdn.txt' % outdir, 'w')
    self.fp_mdn.write('Iteration; Loss; L2 Loss;\n')
    self.fp_mdn.flush()

  def update_plot(self, x, losses, plot_type='vae'):

    if(plot_type == 'vae'):
      self.fp_vae.write('%f;' % x)
      for loss_i, loss in enumerate(losses):
        win = self.vis_plot_vae[loss_i]
        self.vis.updateTrace(
          X=np.array([x], dtype='f'),
          Y=np.array([loss], dtype='f'),
          win=win)
        self.fp_vae.write(' %f;' % loss)
      self.fp_vae.write('\n')
      self.fp_vae.flush()
        

    elif(plot_type == 'mdn'):
      for loss_i, loss in enumerate(losses):
        win = self.vis_plot_mdn[loss_i]
        self.vis.updateTrace(
          X=np.array([x], dtype='f'),
          Y=np.array([losses[loss_i]], dtype='f'),
          win=win)
      self.fp_mdn.write('%f; %f; %f;\n' % (x, losses[0], losses[1]))
      self.fp_mdn.flush()

  def update_test_plot(self, x, y):
    self.vis.updateTrace(
      X=np.array([x], dtype='f'),
      Y=np.array([y], dtype='f'),
      win=self.vis_plot_test_vae)
    self.fp_test_vae.write('%f; %f;\n' % (x, y))
    self.fp_test_vae.flush()

