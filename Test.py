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
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from tqdm import tqdm
from data_loader import data_loader
from gain import gain
from Egain import Egain
from utils import rmse_loss
import xlwt
def main ():
    data_names = ['letter', 'spam']
    # data_names = ['breasttissue','glass', 'thyroid']
    # data with continuous feature and not originally missing

    # data_names = ['balance','banknote','blood','breasttissue', 'climate','connectionistvowel',
    #               'ecoli','glass','hillvalley','ionosphere', 'parkinsons','planning','seedst',
    #               'thyroid','vehicle','vertebral','wine','yeast']
    print(len(data_names))
    miss_rate = 0.2
    batch_size = 64
    alpha = 100
    iterations = 1000
    n_times = 30

    wb_gain = xlwt.Workbook()
    sh_rmse_gain = wb_gain.add_sheet("GAIN_rmse")
    sh_acc_dct_gain = wb_gain.add_sheet("GAIN_acc_dct")
    sh_acc_knn_gain = wb_gain.add_sheet("GAIN_acc_knn")
    sh_acc_nb_gain = wb_gain.add_sheet("GAIN_acc_nb")
    sh_acc_lr_gain = wb_gain.add_sheet("GAIN_acc_lr")

    wb_egain = xlwt.Workbook()
    sh_rmse_egain = wb_egain .add_sheet("EGAIN_rmse")
    sh_acc_dct_egain = wb_egain.add_sheet("EGAIN_acc_dct")
    sh_acc_knn_egain = wb_egain.add_sheet("EGAIN_acc_knn")
    sh_acc_nb_egain = wb_egain.add_sheet("EGAIN_acc_nb")
    sh_acc_lr_egain = wb_egain.add_sheet("EGAIN_acc_lr")

    wb_mean = xlwt.Workbook()
    sh_rmse_mean = wb_mean.add_sheet("MEAN_rmse")
    sh_acc_dct_mean = wb_mean.add_sheet("MEAN_acc_dct")
    sh_acc_knn_mean = wb_mean.add_sheet("MEAN_acc_knn")
    sh_acc_nb_mean = wb_mean.add_sheet("MEAN_acc_nb")
    sh_acc_lr_mean = wb_mean.add_sheet("MEAN_acc_lr")

    wb_knn = xlwt.Workbook()
    sh_rmse_knn = wb_knn.add_sheet("KNN_rmse")
    sh_acc_dct_knn = wb_knn.add_sheet("KNN_acc_dct")
    sh_acc_knn_knn = wb_knn.add_sheet("KNN_acc_knn")
    sh_acc_nb_knn = wb_knn.add_sheet("KNN_acc_nb")
    sh_acc_lr_knn = wb_knn.add_sheet("KNN_acc_lr")

    for k in range(len(data_names)):

        data_name = data_names[k]
        gain_parameters = {'batch_size': batch_size,
                         'alpha': alpha,
                         'iterations': iterations}
        ori_data_x, y, miss_data_x, m = data_loader(data_name, miss_rate)

        print("Dataset: ", data_name)

        ###########################Mean imputation#################################
        print('Mean imputation')
        sh_rmse_mean.write(0, k, data_name)
        sh_acc_dct_mean.write(0, k, data_name)
        sh_acc_knn_mean.write(0, k, data_name)
        sh_acc_nb_mean.write(0, k, data_name)
        sh_acc_lr_mean.write(0, k, data_name)

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputed_data_x = imp.fit_transform(miss_data_x)

        sh_rmse_mean.write(1, k, np.round(rmse_loss(ori_data_x, imputed_data_x, m), 4))

        # Normalize data and classification
        # imputed_data_x, _ = normalization(imputed_data_x)
        #
        # scf = StratifiedShuffleSplit(n_splits=10)
        # # DCT classifier
        # score_dct_mean = cross_val_score(DecisionTreeClassifier(), imputed_data_x, y, cv=scf, scoring='accuracy')
        # sh_acc_dct_mean.write(1, k, np.round(np.mean(score_dct_mean), 4))
        # # KNN classifier
        # score_knn_mean = cross_val_score(KNeighborsClassifier(), imputed_data_x, y, cv=scf, scoring='accuracy')
        # sh_acc_knn_mean.write(1, k, np.round(np.mean(score_knn_mean), 4))
        # # NB classifier
        # score_nb_mean = cross_val_score(GaussianNB(), imputed_data_x, y, cv=scf, scoring='accuracy')
        # sh_acc_nb_mean.write(1, k, np.round(np.mean(score_nb_mean), 4))
        # # LR classifier
        # score_lr_mean = cross_val_score(LogisticRegression(max_iter=1000), imputed_data_x, y, cv=scf,
        #                                 scoring='accuracy')
        # sh_acc_lr_mean.write(1, k, np.round(np.mean(score_lr_mean), 4))

        ###########################KNN imputation#################################
        print('KNN imputation')
        sh_rmse_knn.write(0, k, data_name)
        sh_acc_dct_knn.write(0, k, data_name)
        sh_acc_knn_knn.write(0, k, data_name)
        sh_acc_nb_knn.write(0, k, data_name)
        sh_acc_lr_knn.write(0, k, data_name)

        imp = KNNImputer(missing_values=np.nan)
        imputed_data_x = imp.fit_transform(miss_data_x)

        sh_rmse_knn.write(1, k, np.round(rmse_loss(ori_data_x, imputed_data_x, m), 4))

        # # Normalize data and classification
        # imputed_data_x, _ = normalization(imputed_data_x)
        #
        # scf = StratifiedShuffleSplit(n_splits=10)
        # # DCT classifier
        # score_dct_knn = cross_val_score(DecisionTreeClassifier(), imputed_data_x, y, cv=scf, scoring='accuracy')
        # sh_acc_dct_knn.write(1, k, np.round(np.mean(score_dct_knn), 4))
        # # KNN classifier
        # score_knn_knn = cross_val_score(KNeighborsClassifier(), imputed_data_x, y, cv=scf, scoring='accuracy')
        # sh_acc_knn_knn.write(1, k, np.round(np.mean(score_knn_knn), 4))
        # # NB classifier
        # score_nb_knn = cross_val_score(GaussianNB(), imputed_data_x, y, cv=scf, scoring='accuracy')
        # sh_acc_nb_knn.write(1, k, np.round(np.mean(score_nb_knn), 4))
        # # LR classifier
        # score_lr_knn = cross_val_score(LogisticRegression(max_iter=1000), imputed_data_x, y, cv=scf,
        #                                 scoring='accuracy')
        # sh_acc_lr_knn.write(1, k, np.round(np.mean(score_lr_knn), 4))

        ###########################GAIN imputation#################################
        print('GAIN imputation')
        sh_rmse_gain.write(0,k,data_name)
        sh_acc_dct_gain.write(0,k,data_name)
        sh_acc_knn_gain.write(0,k,data_name)
        sh_acc_nb_gain.write(0,k,data_name)
        sh_acc_lr_gain.write(0,k,data_name)
        for i in tqdm(range(n_times)):
            # Impute missing data
            imputed_data_x = gain(miss_data_x, gain_parameters)
            sh_rmse_gain.write(i + 1, k, np.round(rmse_loss(ori_data_x, imputed_data_x, m), 4))

            # #Normalize data and classification
            # imputed_data_x,_ = normalization(imputed_data_x)
            #
            # scf = StratifiedShuffleSplit(n_splits=10)
            # #DCT classifier
            # score_dct_gain = cross_val_score(DecisionTreeClassifier(),imputed_data_x, y, cv=scf, scoring='accuracy')
            # sh_acc_dct_gain.write(i+1, k, np.round(np.mean(score_dct_gain), 4))
            # #KNN classifier
            # score_knn_gain = cross_val_score(KNeighborsClassifier(),imputed_data_x, y, cv=scf, scoring='accuracy')
            # sh_acc_knn_gain.write(i+1, k, np.round(np.mean(score_knn_gain), 4))
            # #NB classifier
            # score_nb_gain = cross_val_score(GaussianNB(),imputed_data_x, y, cv=scf, scoring='accuracy')
            # sh_acc_nb_gain.write(i+1, k, np.round(np.mean(score_nb_gain), 4))
            # #LR classifier
            # score_lr_gain = cross_val_score(LogisticRegression(max_iter=1000),imputed_data_x, y, cv=scf, scoring='accuracy')
            # sh_acc_lr_gain.write(i+1, k, np.round(np.mean(score_lr_gain), 4))

        ###########################EGAIN imputation#################################
        print('EGAIN imputation')
        sh_rmse_egain.write(0,k,data_name)
        sh_acc_dct_egain.write(0,k,data_name)
        sh_acc_knn_egain.write(0,k,data_name)
        sh_acc_nb_egain.write(0,k,data_name)
        sh_acc_lr_egain.write(0,k,data_name)

        for i in tqdm(range(n_times)):

            imputed_data_x = Egain(miss_data_x, gain_parameters)
            sh_rmse_egain.write(i + 1, k, np.round(rmse_loss(ori_data_x, imputed_data_x, m), 4))

            # Normalize data and classification
            # imputed_data_x, _ = normalization(imputed_data_x)
            #
            # scf = StratifiedShuffleSplit(n_splits=10)
            # # DCT classifier
            # score_dct_egain = cross_val_score(DecisionTreeClassifier(), imputed_data_x, y, cv=scf, scoring='accuracy')
            # sh_acc_dct_egain.write(i + 1, k, np.round(np.mean(score_dct_egain), 4))
            # # KNN classifier
            # score_knn_egain = cross_val_score(KNeighborsClassifier(), imputed_data_x, y, cv=scf, scoring='accuracy')
            # sh_acc_knn_egain.write(i + 1, k, np.round(np.mean(score_knn_egain), 4))
            # # NB classifier
            # score_nb_egain = cross_val_score(GaussianNB(), imputed_data_x, y, cv=scf, scoring='accuracy')
            # sh_acc_nb_egain.write(i + 1, k, np.round(np.mean(score_nb_egain), 4))
            # # LR classifier
            # score_lr_egain = cross_val_score(LogisticRegression(max_iter=1000), imputed_data_x, y, cv=scf,
            #                                 scoring='accuracy')
            # sh_acc_lr_egain.write(i + 1, k, np.round(np.mean(score_lr_egain), 4))

    wb_gain.save('GAIN_test.xls')
    wb_egain.save('EGAIN_test.xls')
    wb_mean.save('MEAN_test.xls')
    wb_knn.save('KNN_test.xls')
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
