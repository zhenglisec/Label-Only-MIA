'''
References:
[1] J. Cohen, E. Rosenfeld and Z. Kolter. 
Certified Adversarial Robustness via Randomized Smoothing. In ICML, 2019.

Acknowledgements:
[1] https://github.com/locuslab/smoothing/blob/master/code/certify.py
'''

import numpy as np
#from PIL import Image
#import scipy.io as sio

from cert_radius.core import Smooth
from runx.logx import logx

def certify(model, device, dataset, transform, num_classes, matfile=None,
            mode='hard', start_img=0, num_img=500, skip=1, sigma=0.25, N0=100, N=100000,
            alpha=0.001, batch=1000, verbose=False,
            grid=(0.1, 0.15, 0.2, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5), beta=1.0):
  logx.msg('===certify(N={}, sigma={}, mode={})==='.format(N, sigma, mode))

  model.eval()
  smoothed_net = Smooth(model, num_classes,
                        sigma, device, mode, beta)

  radius_hard = np.zeros((num_img,), dtype=np.float)
  radius_soft = np.zeros((num_img,), dtype=np.float)
  num_grid = len(grid)
  cnt_grid_hard = np.zeros((num_grid + 1,), dtype=np.int)
  cnt_grid_soft = np.zeros((num_grid + 1,), dtype=np.int)
  s_hard = 0.0
  s_soft = 0.0
  
  for i in range(num_img):
    print(i)
    #import time
    #time.sleep(120)
    img, target = dataset[start_img + i * skip]
    img = img.to(device)

    if mode == 'both':
      p_hard, r_hard, p_soft, r_soft = smoothed_net.certify(
          img, N0, N, alpha, batch)
      correct_hard = int(p_hard == target)
      correct_soft = int(p_soft == target)
      if verbose:
        if correct_hard == 1:
          logx.msg('Hard Correct: 1. Radius: {}.'.format(r_hard))
        else:
          logx.msg('Hard Correct: 0.')
        if correct_soft == 1:
          logx.msg('Soft Correct: 1. Radius: {}.'.format(r_soft))
        else:
          logx.msg('Soft Correct: 0.')
      radius_hard[i] = r_hard if correct_hard == 1 else -1
      radius_soft[i] = r_soft if correct_soft == 1 else -1
      if correct_hard == 1:
        cnt_grid_hard[0] += 1
        s_hard += r_hard
        for j in range(num_grid):
          if r_hard >= grid[j]:
            cnt_grid_hard[j + 1] += 1
      if correct_soft == 1:
        cnt_grid_soft[0] += 1
        s_soft += r_soft
        for j in range(num_grid):
          if r_soft >= grid[j]:
            cnt_grid_soft[j + 1] += 1
    else:
      prediction, radius = smoothed_net.certify(img, N0, N, alpha, batch)
      correct = int(prediction == target)
      if verbose:
        if correct == 1:
          logx.msg('Correct: 1. Radius: {}.'.format(radius))
        else:
          logx.msg('Correct: 0.')
      radius_hard[i] = radius if correct == 1 else -1
      if correct == 1:
        cnt_grid_hard[0] += 1
        s_hard += radius
        for j in range(num_grid):
          if radius >= grid[j]:
            cnt_grid_hard[j + 1] += 1

  logx.msg('===Certify Summary===')
  logx.msg('Total Image Number: {}'.format(num_img))
  if mode == 'both':
    logx.msg('===Hard certify===')
    logx.msg('Radius: 0.0  Number: {}  Acc: {}'.format(
        cnt_grid_hard[0], cnt_grid_hard[0] / num_img * 100))
    for j in range(num_grid):
      logx.msg('Radius: {}  Number: {}  Acc: {}'.format(
          grid[j], cnt_grid_hard[j + 1], cnt_grid_hard[j + 1] / num_img * 100))
    logx.msg('ACR: {}'.format(s_hard / num_img))
    logx.msg('===Soft certify===')
    logx.msg('Radius: 0.0  Number: {}  Acc: {}'.format(
        cnt_grid_soft[0], cnt_grid_soft[0] / num_img * 100))
    for j in range(num_grid):
      logx.msg('Radius: {}  Number: {}  Acc: {}'.format(
          grid[j], cnt_grid_soft[j + 1], cnt_grid_soft[j + 1] / num_img * 100))
    logx.msg('ACR: {}'.format(s_soft / num_img))
    #if matfile is not None:
      #sio.savemat(matfile, {'hard': radius_hard, 'soft': radius_soft})
  else:
    logx.msg('Radius: 0.0  Number: {}  Acc: {}'.format(
        cnt_grid_hard[0], cnt_grid_hard[0] / num_img * 100))
    for j in range(num_grid):
      logx.msg('Radius: {}  Number: {}  Acc: {}'.format(
          grid[j], cnt_grid_hard[j + 1], cnt_grid_hard[j + 1] / num_img * 100))
    logx.msg('ACR: {}'.format(s_hard / num_img))
    #if matfile is not None:
      #sio.savemat(matfile, {mode: radius_hard})
