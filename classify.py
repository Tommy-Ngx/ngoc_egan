'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from utils import normalization, renormalization, rounding
import argparse
import numpy as np

from data_loader import data_loader
from gain import gain
from utils import rmse_loss
import xlwt
def main ():
    # data_names = ['letter', 'spam']
    # data_names = ['balance']
    # data with continuous feature and not originally missing
    #
    data_names = ['balance','banknote','blood','breasttissue', 'climate','connectionistvowel',
                  'ecoli','glass','hillvalley','ionosphere', 'parkinsons','planning','seedst',
                  'thyroid','vehicle','vertebral','wine','yeast']
    print(len(data_names))
    miss_rate = 0.2
    batch_size = 64
    alpha = 100
    iterations = 1000
    n_times = 30

    for k in range(len(data_names)):

        data_name = data_names[k]

        print("Dataset: ", data_name)
        rmse = []
        # acc_dct = []
        # acc_knn = []
        # acc_nb = []
        ori_data_x, y, _, _ = data_loader(data_name, miss_rate)
        ori_data_x, _ = normalization(ori_data_x)
        scf = StratifiedShuffleSplit(n_splits=5)
        score_dct = cross_val_score(DecisionTreeClassifier(max_depth=5),ori_data_x, y, cv=scf, scoring='accuracy')
        print(score_dct)
        score_knn = cross_val_score(KNeighborsClassifier(),ori_data_x, y, cv=scf, scoring='accuracy')
        print(score_knn)
        score_nb = cross_val_score(GaussianNB(),ori_data_x, y, cv=scf, scoring='accuracy')
        print(score_nb)

        print("---------------------------")
if __name__ == '__main__':
    main()
    # Inputs for the main function
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #   '--data_name',
    #   choices=['letter','spam'],
    #   default='spam',
    #   type=str)
    # parser.add_argument(
    #   '--miss_rate',
    #   help='missing data probability',
    #   default=0.2,
    #   type=float)
    # parser.add_argument(
    #   '--batch_size',
    #   help='the number of samples in mini-batch',
    #   default=128,
    #   type=int)
    # parser.add_argument(
    #   '--alpha',
    #   help='hyperparameter',
    #   default=10,
    #   type=float)
    # parser.add_argument(
    #   '--iterations',
    #   help='number of training interations',
    #   default=5000,
    #   type=int)
    #
    # args = parser.parse_args()
    #
    # # Calls main function
    # main(args)
