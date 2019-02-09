# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
#import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from Feature_extraction import *
from scipy import sparse, io
from sklearn.decomposition import PCA
from preprocessing_data import split_data
from SVM_functions import *
from preprocessing_data import dimensionality_reduction

# Utility function to move the midpoint of a colormap to be around the values of interest.
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class TrainerLinear:
    def __init__(self, training_data, training_target, test_data, test_target):
        self.training_data = training_data
        self.training_target = training_target
        self.test_data = test_data
        self.test_target = test_target
        self.clf = svm.SVC(C=1, class_weight="balanced", coef0=0.0,
                           decision_function_shape=None, degree=3, gamma='auto',
                           kernel='linear', max_iter=-1, probability=False,
                           random_state=None, shrinking=True, tol=0.001, verbose=False)

    def learn_best_param(self):
        print "----------正在寻找最优参数----------"
        C_range = np.logspace(-2, 10, 13)
        param_grid = dict(C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(self.training_data, self.training_target)
        self.clf.set_params(C=grid.best_params_['C'])
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

    def train_classifier(self):
        print "----------正在训练线性SVM模型----------"
        self.clf.fit(self.training_data, self.training_target)
        joblib.dump(self.clf, '../model/SVM_linear_estimator.pkl')
        print "模型训练完毕，已成功保存！"
        #training_result = self.clf.predict(self.training_data)
        #print metrics.classification_report(self.training_target, training_result)

    def load_model(self):
        print "----------正在加载已保存的模型----------"
        self.clf = joblib.load('../model/SVM_linear_estimator.pkl')

    def predict_test_data(self):
        print "----------正在测试模型----------"
        test_results = self.clf.predict(self.test_data)
        print metrics.classification_report(self.test_target, test_results)
        print metrics.confusion_matrix(self.test_target, test_results)

    def cross_validation(self):
        print "----------正在对训练集进行交叉验证----------"
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=20)
        scores = cross_val_score(self.clf, self.training_data, self.training_target, cv=cv, scoring='f1_macro')
        print scores
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


class TrainerRbf:
    def __init__(self, training_data, training_target, test_data, test_target):
        self.training_data = training_data
        self.training_target = training_target
        self.test_data = test_data
        self.test_target = test_target
        self.clf = svm.SVC(C=100, class_weight=None, coef0=0.0,
                           decision_function_shape=None, degree=3, gamma=0.01,
                           kernel='rbf', max_iter=-1, probability=False,
                           random_state=None, shrinking=True, tol=0.001, verbose=False)

    def learn_best_param(self):
        print "----------正在寻找最优参数----------"
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(self.training_data, self.training_target)
        self.clf.set_params(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        #self.draw_visualization_param_effect(grid, C_range, gamma_range)

    '''def draw_visualization_param_effect(self, grid, C_range, gamma_range):
        scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                             len(gamma_range))
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest',
                   norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        plt.savefig('fig/param_effect.png')
        plt.show()'''

    def train_classifier(self):
        print "----------正在训练RBF核SVM模型----------"
        self.clf.fit(self.training_data, self.training_target)
        joblib.dump(self.clf, '../model/SVM_rbf_estimator.pkl')
        print "模型训练完毕，已成功保存！"
        #training_result = self.clf.predict(self.training_data)
        #print metrics.classification_report(self.training_target, training_result)

    def load_model(self):
        print "----------正在加载已保存的模型----------"
        self.clf = joblib.load('../model/SVM_linear_estimator.pkl')

    def predict_test_data(self):
        print "----------正在测试模型----------"
        test_results = self.clf.predict(self.test_data)
        print metrics.classification_report(self.test_target, test_results)
        print metrics.confusion_matrix(self.test_target, test_results)


    def cross_validation(self):
        print "----------正在对训练集进行交叉验证----------"
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=20)
        scores = cross_val_score(self.clf, self.training_data, self.training_target, cv=cv, scoring='f1_macro')
        print scores
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def count_time(time0):
    time1 = time.time()
    t = time1 - time0
    print "当前用时为： ",t


if '__main__' == __name__:

    time0 = time.time()

    label = load_class_data("../data/derived/classLabel.txt")
    content = load_train_vectors(5000,3006,"../model/train_vectors.txt")

    training_data, test_data, training_target, test_target = train_test_split(content, label, test_size=0.1) #划分训练集和测试集
    #training_data, test_data = dimensionality_reduction(training_data.todense(), test_data.todense(), type='pca')


    Trainer = TrainerLinear(training_data, training_target, test_data, test_target)
    Trainer.learn_best_param()
    count_time(time0)
    Trainer.train_classifier()
    count_time(time0)
    #Trainer.load_model()
    #count_time(time0)
    Trainer.predict_test_data()
    count_time(time0)
    Trainer.cross_validation() #仅对训练集进行的交叉验证！
    count_time(time0)
    print 'linear ok'

    '''
    Trainer2 = TrainerRbf(training_data, training_target, test_data, test_target)
    Trainer2.learn_best_param()
    count_time(time0)
    Trainer2.train_classifier()
    count_time(time0)
    #Trainer2.load_model()
    #count_time(time0)
    Trainer2.predict_test_data()
    count_time(time0)
    Trainer2.cross_validation()
    count_time(time0)
    print "rbf ok"
    '''


    time1 = time.time()
    t = time1 - time0
    print "总用时为： ",t


