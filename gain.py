'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
import copy

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index, sample_batch_binary


def gain (miss_data_x, gain_parameters):
  '''Impute missing values in data_x
  
  Args:
    - miss_data_x: missing data
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  m = 1-np.isnan(miss_data_x)
  
  # System parameters
  batch_size = gain_parameters['batch_size']
  # hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  
  # Other parameters
  no, dim = miss_data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(miss_data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  
  ## GAIN architecture
  tf.reset_default_graph()
  # Input placeholders
  # Data vector
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector 
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # # Hint vector
  # H = tf.placeholder(tf.float32, shape = [None, dim])
  # B vector
  B = tf.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## GAIN structure


  # Generator
  G_sample = generator(X, M)
  H = B*M + 0.5*(1-B)
  D_prob_g = discriminator(X * M + G_sample * (1 - M), H)

  fake_X = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  Hat_X = X * M + fake_X * (1 - M)
  # Discriminator
  D_prob = discriminator(Hat_X, H)

  # GAIN loss
  # D_loss_temp = -tf.reduce_mean((1-B)*(M * tf.log(D_prob + 1e-8) \
  #                               + (1-M) * tf.log(1. - D_prob + 1e-8))) \
  #                               / tf.reduce_mean(1-B)
  #
  # G_loss_temp = -tf.reduce_mean((1-B)*(1-M) * tf.log(D_prob + 1e-8)) / tf.reduce_mean(1-B)
  D_loss_temp = -tf.reduce_mean((M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)))

  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob_g + 1e-8))
  MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  
  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss 
  
  ## GAIN solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  gen_new_params = []
  params = []
  for param in theta_G:
    params.append(sess.run(param))
  gen_new_params.append(params)

  for it in range(iterations):
  # for it in tqdm(range(iterations)):
    # Sample batch
    # print(sess.run(theta_G[-1]))
    gen_old_params = copy.deepcopy(gen_new_params)
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]
    M_mb = m[batch_idx, :]
    # Sample random vectors  
    Z_mb = uniform_sampler(0.0, 0.01, batch_size, dim)
    # Sample hint vectors
    # H_mb_temp = binary_sampler(0.9, batch_size, dim)
    # H_mb = M_mb * H_mb_temp
    # H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    B_mb = sample_batch_binary(dim, batch_size)
    # H_mb = B_mb*M_mb + 0.5*(1-B_mb)

    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
    f_mb = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
    # print(f_mb)
    for w in range(len(theta_G)):
      theta_G[w].load(gen_new_params[0][w], sess)
    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              feed_dict = {X:X_mb, M:M_mb, fake_X:f_mb, B:B_mb})

    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]
    M_mb = m[batch_idx, :]
    # Sample random vectors
    Z_mb = uniform_sampler(0.0, 0.01, batch_size, dim)
    # Sample hint vectors
    # H_mb_temp = binary_sampler(0.9, batch_size, dim)
    # H_mb = M_mb * H_mb_temp
    # H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    B_mb = sample_batch_binary(dim, batch_size)
    # H_mb = B_mb*M_mb + 0.5*(1-B_mb)

    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
    for w in range(len(theta_G)):
      theta_G[w].load(gen_old_params[0][w], sess)
    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X_mb, M: M_mb, B: B_mb})
    params = []
    for param in theta_G:
      params.append(sess.run(param))
    gen_new_params[0] = params
  ## Return imputed data
  Z_mb = uniform_sampler(0.0, 0.01, no, dim)
  M_mb = m
  X_mb = norm_data_x
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
  for w in range(len(theta_G)):
    theta_G[w].load(gen_new_params[0][w], sess)
  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
  sess.close()
  imputed_data = m * norm_data_x + (1-m) * imputed_data
  
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)
  
  # Rounding
  imputed_data = rounding(imputed_data, miss_data_x)

  return imputed_data