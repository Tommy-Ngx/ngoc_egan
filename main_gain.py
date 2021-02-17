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
from sklearn.linear_model import LogisticRegression
from utils import normalization, renormalization, rounding
import argparse
import numpy as np

from data_loader import data_loader
from gain import gain
from utils import rmse_loss
import xlwt
def main ():
    # data_names = ['letter', 'spam']
    # data_names = ['breasttissue','glass', 'thyroid']
    # data with continuous feature and not originally missing

    data_names = ['balance','banknote','blood','breasttissue', 'climate','connectionistvowel',
                  'ecoli','glass','hillvalley','ionosphere', 'parkinsons','planning','seedst',
                  'thyroid','vehicle','vertebral','wine','yeast']
    print(len(data_names))
    miss_rate = 0.15
    batch_size = 64
    alpha = 100
    iterations = 1000
    n_times = 30
    wb = xlwt.Workbook()
    sh_rmse = wb.add_sheet("GAIN_rmse")
    # sh_acc = wb.add_sheet("EGAIN_acc")
    sh_acc_dct = wb.add_sheet("GAIN_acc_dct")
    sh_acc_knn = wb.add_sheet("GAIN_acc_knn")
    sh_acc_nb = wb.add_sheet("GAIN_acc_nb")
    sh_acc_lr = wb.add_sheet("GAIN_acc_lr")

    for k in range(len(data_names)):

        data_name = data_names[k]
        gain_parameters = {'batch_size': batch_size,
                         'alpha': alpha,
                         'iterations': iterations}
        print("Dataset: ", data_name)
        rmse = []
        # acc_dct = []
        # acc_knn = []
        # acc_nb = []
        ori_data_x, y, miss_data_x, m = data_loader(data_name, miss_rate)
        sh_rmse.write(0,k,data_name)
        sh_acc_dct.write(0,k,data_name)
        sh_acc_knn.write(0,k,data_name)
        sh_acc_nb.write(0,k,data_name)
        sh_acc_lr.write(0, k, data_name)
        # sh_acc.write(0, 0, 'dct')
        # sh_acc.write(0, 1, 'knn')
        # sh_acc.write(0, 2, 'nb')
        for i in range(n_times):

            # Impute missing data
            imputed_data_x = gain(miss_data_x, gain_parameters)
            imputed_data_x,_ = normalization(imputed_data_x)

            # Calculate rmse
            rmse.append(rmse_loss (ori_data_x, imputed_data_x, m))

            print('{:2d}/{:2d}'.format(i+1,n_times), end=':')
            print('RMSE = ' + str(np.round(rmse[-1], 4)))
            sh_rmse.write(i+1,k,str(np.round(rmse[-1], 4)))
            if data_name in ['letter', 'spam']:
                continue
            scf = StratifiedShuffleSplit(n_splits=10)
            score_dct = cross_val_score(DecisionTreeClassifier(),imputed_data_x, y, cv=scf, scoring='accuracy')
            print(score_dct)
            # acc_dct.extend(score_dct)
            sh_acc_dct.write(i+1, k, str(np.round(np.mean(score_dct), 4)))
            # for j in range(len(score_dct)):
                # sh_acc.write(i * 5 + j + 1, 0, str(np.round(score_dct[j], 4)))
            score_knn = cross_val_score(KNeighborsClassifier(),imputed_data_x, y, cv=scf, scoring='accuracy')
            print(score_knn)
            # acc_knn.extend(score_knn)
            sh_acc_knn.write(i+1, k, str(np.round(np.mean(score_knn), 4)))
            # for j in range(len(score_knn)):
                # sh_acc.write(i * 5 + j + 1, 1, str(np.round(score_knn[j], 4)))
            score_nb = cross_val_score(GaussianNB(),imputed_data_x, y, cv=scf, scoring='accuracy')
            print(score_nb)
            # acc_nb.extend(score_nb)
            sh_acc_nb.write(i+1, k, str(np.round(np.mean(score_nb), 4)))
            # for j in range(len(score_nb)):
                # sh_acc.write(i * 5 + j + 1, 2, str(np.round(score_nb[j], 4)))
            score_lr = cross_val_score(LogisticRegression(max_iter=1000),imputed_data_x, y, cv=scf, scoring='accuracy')
            print(score_lr)
            # acc_nb.extend(score_nb)
            sh_acc_lr.write(i+1, k, str(np.round(np.mean(score_lr), 4)))
        # rmse = np.array(rmse)
        # acc_dct = np.array(acc_dct)
        # acc_knn = np.array(acc_knn)
        # acc_nb = np.array(acc_nb)
        # print("RMSE mean = {:.4f}; variance = {:.4f} ".format(np.mean(rmse), np.std(rmse)))
        # print("Acc mean = {:.4f}; variance = {:.4f} ".format(np.mean(acc_dct), np.std(acc_dct)))
        # print("Acc mean = {:.4f}; variance = {:.4f} ".format(np.mean(acc_knn), np.std(acc_knn)))
        # print("Acc mean = {:.4f}; variance = {:.4f} ".format(np.mean(acc_nb), np.std(acc_nb)))
        print("---------------------------")
    wb.save('GAIN_results_15.xls')
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
