'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
from utils import binary_sampler
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
# from keras.datasets import mnist


# def data_loader (data_name, miss_rate):
#   '''Loads datasets and introduce missingness.
#
#   Args:
#     - data_name: letter, spam, or mnist
#     - miss_rate: the probability of missing components
#
#   Returns:
#     data_x: original data
#     miss_data_x: data with missing values
#     data_m: indicator matrix for missing components
#   '''
#
#   # Load data
#   if data_name in ['letter', 'spam']:
#     file_name = 'data/'+data_name+'.csv'
#     data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
#   elif data_name == 'mnist':
#     (data_x, _), _ = mnist.load_data()
#     data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)
#
#   # Parameters
#   no, dim = data_x.shape
#
#   # Introduce missing data
#   data_m = binary_sampler(1-miss_rate, no, dim)
#   miss_data_x = data_x.copy()
#   miss_data_x[data_m == 0] = np.nan
#
#   return data_x, miss_data_x, data_m

def data_loader(data_name, miss_rate):
  if data_name in ['spam', 'letter']:
    file_name = 'data/' + data_name + '.csv'
    x = np.loadtxt(file_name, delimiter=",", skiprows=1)
    y = []
  else:
    file_name = 'data/' + data_name + '.arff'
    data, _ = arff.loadarff(file_name)
    data = data.tolist()
    x = np.array([item[:-1] for item in data])
    y = np.array([item[-1] for item in data])
    le = LabelEncoder()
    y = le.fit_transform(y)
    print('Num of classes: ', len(le.classes_))
  # Parameters
  no, dim = x.shape
  print('Num of samples:', no)
  print('Num of features: ', dim)
  print(type(x[0][0]))
  # Introduce missing data
  m = binary_sampler(1 - miss_rate, no, dim)
  miss_x = x.copy()
  miss_x[m == 0] = np.nan

  return x, y, miss_x, m