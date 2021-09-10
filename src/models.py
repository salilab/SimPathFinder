import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from gensim.models.fasttext import FastText as FT_gensim
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, KFold
from sklearn.metrics import fbeta_score, f1_score, hamming_loss, accuracy_score, recall_score, confusion_matrix, precision_score, log_loss, classification_report, mean_squared_error, make_scorer, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import seaborn as sns


class ModelVectors(object):
    def __init__(self, data_dir='../labeldata/',
                 model_dir_emb='../models/',
                 categories=['Detoxification',
                             'Activation',
                                'Biosynthesis',
                             'Degradation',
                             'Energy',
                             'Glycan',
                                'Macromolecule'],
                 modelName='tierT12_10_300'):
        super(ModelVectors, self).__init__()
        self.data_dir = data_dir
        self.model_dir_emb = model_dir_emb
        self.categories = categories
        self.modelName = modelName
        self.train = pickle.load(
            open(self.data_dir + 'TrainLabeledData.pkl', "rb"))
        self.validate = pickle.load(
            open(self.data_dir + 'ValidateLabeledData.pkl', "rb"))
        self.model = FT_gensim.load(self.model_dir + self.modelName+'.model')
        self.trainLSTM = pickle.load(
            open(self.data_dir + 'TrainLabeledDataLSTM.pkl', "rb"))
        self.validateLSTM = pickle.load(
            open(self.data_dir + 'ValidateLabeledDataLSTM.pkl', "rb"))
        self.testLSTM = pickle.load(
            open(self.data_dir + 'TestLabeledDataLSTM.pkl', "rb"))

    def fetchModel(self):
        self.model = FT_gensim.load(self.model_dir + self.modelName+'.model')
        return self.model

    def pwyVector(self, pwy):
        return np.mean(self.model.wv[pwy], axis=0)

    def getXYTrain(self, tag='EC'):
        self.train['pathway_vector'] = self.train[tag].apply(self.pwyVector)
        X_train = list(self.train.pathway_vector)
        Y_train = self.train[self.categories]
        return X_train, Y_train

    def getXYValidate(self, tag='EC'):
        self.validate['pathway_vector'] = self.validate[tag].apply(
            self.pwyVector)
        X_validate = list(self.validate.pathway_vector)
        Y_validate = self.validate[self.categories]
        return X_validate, Y_validate

    def getXYValidateLSTM(self, tag='EC'):
        self.validate['pathway_vector'] = self.validateLSTM[tag].apply(
            self.pwyVector)
        X_validate = list(self.validateLSTM.pathway_vector)
        Y_validate = self.validateLSTM[self.categories]
        return X_validate, Y_validate

    def getControl(self, pwy):
        self.enzymes = self.getAllEnzymes()
        output_lst = []
        for enz in self.enzymes:
            if enz in pwy:
                output_lst.append(1)
            else:
                output_lst.append(0)
        return np.array(output_lst)

    def getAllEnzymes(self):
        self.allData = pickle.load(
            open(self.data_dir + 'LabeledData.pkl', "rb"))
        self.enzymes = set([j for i in self.allData.EC for j in i])
        return self.enzymes

    def getXYControl(self):
        self.train['control_vector'] = self.train.EC.apply(self.getControl)
        self.validate['control_vector'] = self.validate.EC.apply(
            self.getControl)
        XC_train = list(self.train.control_vector)
        XC_validate = list(self.validate.control_vector)
        Y_validate = self.validate[self.categories]
        Y_train = self.train[self.categories]
        return XC_train, Y_train, XC_validate, Y_validate


class BuildClassicalModel(ModelVectors):
    def __init__(self):
        super(BuildClassicalModel, self).__init__(data_dir='labeldata/',
                                                  model_dir='models/',
                                                  categories=['Detoxification',
                                                              'Activation',
                                                              'Biosynthesis',
                                                              'Degradation',
                                                              'Energy',
                                                              'Glycan',
                                                              'Macromolecule'])
        self.fetchModel()
        self.X_train, self.Y_train = self.getXYTrain()
        self.X_valid, self.Y_valid = self.getXYValidate()

    def generateSampleWeights(self, ):
        self.sample_weights = np.array([1 if sum(self.Y_train.values[i, :]) == 1 else sum(self.Y_train.values[i, :])
                                        for i in range(self.Y_train.values.shape[0])])
        return self.sample_weights

    def defineRM(self, criterion='entropy', n_estimators=5000, class_weight='balanced_subsample',):
        self.RMmodel = OneVsRestClassifier(RandomForestClassifier(criterion=criterion,
                                                                  class_weight=class_weight,
                                                                  max_features='log2',
                                                                  n_estimators=n_estimators))

    def fitRM(self, X_train=[], Y_train=[]):
        if not X_train:
            X_train = self.X_train
        if not Y_train:
            Y_train = self.Y_train
        self.defineRM()
        self.generateSampleWeights()
        self.RMmodel.fit(X_train, Y_train)
        return self.RMmodel

    def fitRMControl(self, X_train=[], Y_train=[], n_estimators=500):
        if not X_train:
            X_train = self.X_train
        if not Y_train:
            Y_train = self.Y_train
        self.defineRM(n_estimators=500)
        self.RMmodel.fit(X_train, Y_train)
        print(self.RMmodel.score(X_train, Y_train))
        return self.RMmodel

    def kfoldRM(self):
        self.defineRM()
        kfold = KFold(n_splits=10, shuffle=True, random_state=1)
        scoring = {'Hamming': make_scorer(hamming_loss),
                   'Accuracy': make_scorer(accuracy_score)}
        cv_results = cross_validate(self.RMmodel, self.X_train, self.Y_train,
                                    cv=kfold, scoring=scoring, return_train_score=True)
        accuracy = cv_results['test_Accuracy']
        loss = cv_results['test_Hamming']
        return accuracy, loss

    def spotCheckDefine(self):
        self.models = []
        self.models.append(('SVM-1', OneVsRestClassifier(SVC(probability=True,
                                                             class_weight='balanced',
                                                             kernel='poly'))))
        self.models.append(('SVM-2', OneVsRestClassifier(SVC(probability=True,
                                                             class_weight='balanced',
                                                             kernel='rbf'))))
        self.models.append(('RM', OneVsRestClassifier(RandomForestClassifier(criterion='entropy',
                                                                             class_weight='balanced'))))
        self.models.append(('LR', OneVsRestClassifier(LogisticRegression(class_weight='balanced',
                                                                         solver='newton-cg',
                                                                         max_iter=5000))))
        return self.models

    def spotCheckFit(self):
        self.spotCheckDefine()
        for label, model in self.models:
            model.fit(self.X_train, self.Y_train)
        return

    def spotStrplot(self):
        self.spotCheckFit()
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        scoring = {'Hamming': make_scorer(hamming_loss),
                   'Accuracy': make_scorer(accuracy_score)}
        self.accuracy = []
        self.loss = []
        self.title = []
        for label, model in self.models:
            cv_results = cross_validate(model, self.X_train, self.Y_train,
                                        cv=kfold, scoring=scoring, return_train_score=True)
            self.accuracy.append(cv_results['test_Accuracy'])
            self.loss.append(cv_results['test_Hamming'])
            self.title.append(label)

        self.strplot(self.accuracy, self.title,
                     metricLabel='Exact match accuracy')
        self.strplot(self.loss, self.title, metricLabel='Hamming loss')

    def strplot(self, metricArray, title='', metricLabel='', ylim=[0, 0.75]):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        ax = sns.boxplot(data=metricArray)
        ax = sns.stripplot(data=metricArray,
                           size=8, jitter=True, edgecolor="gray", linewidth=2)
        ax.set_ylabel(metricLabel, fontsize=20)
        ax.set_xticklabels(title)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=16)

    def hyperParameterSearch(self):
        param_grid = {"estimator__n_estimators": [500, 1000, 2000, 3000, 4000, 5000],
                      "estimator__class_weight": ['balanced', 'balanced_subsample'],
                      "estimator__max_features": ['auto', 'sqrt', 'log2']}

        RM = OneVsRestClassifier(RandomForestClassifier())
        grid_search = GridSearchCV(estimator=RM,
                                   param_grid=param_grid,
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=2)
        grid_search.fit(self.X_train, self.Y_train,)
        grid_search.best_params_
        print(grid_search.best_params_)
        return grid_search.best_params_

    def saveModel(self,name='RMmodel.pkl'):
        self.fitRM()
        # modelPkl = pickle.dumps(self.RMmodel)
        with open(self.model_dir+name, 'wb') as f:
            pickle.dump(self.RMmodel, f)


class BuildControlModels(BuildClassicalModel):
    def __init__(self):
        super(BuildControlModels, self).__init__()
        self.X_train, self.Y_train, self.X_valid, self.Y_valid = self.getXYControl()

    def defineLR(self, solver='newton-cg', max_iter=1000, C=10):
        self.LRmodel = OneVsRestClassifier(LogisticRegression(class_weight='balanced',
                                                              solver='newton-cg',
                                                              C=C,
                                                              max_iter=max_iter, penalty='l2'))

    def hyperParameterSearch(self):
        param_grid = {"estimator__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                      "estimator__max_iter": [1000, 2000, 3000, 4000, 5000]}

        LR = OneVsRestClassifier(LogisticRegression(penalty='l2'))
        grid_search = GridSearchCV(estimator=LR,
                                   param_grid=param_grid,
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=2)
        grid_search.fit(self.X_train, self.Y_train)
        grid_search.best_params_
        print(grid_search.best_params_)
        return grid_search.best_params_

    def fitLRControl(self, X_train=[], Y_train=[]):
        if not X_train:
            X_train = self.X_train
        if not Y_train:
            Y_train = self.Y_train
        self.defineLR()
        self.LRmodel.fit(X_train, Y_train)
        print(self.LRmodel.score(X_train, Y_train))
        return self.LRmodel

    def saveModel(self):
        self.fitLRControl()
        with open(self.model_dir+'LRCmodel.pkl', 'wb') as f:
            pickle.dump(self.LRmodel, f)


class BuildAnnot3Models(BuildClassicalModel):
    def __init__(self):
        super(BuildAnnot3Models, self).__init__()
        self.X_train, self.Y_train = self.getXYTrain(tag='annot3')
        self.X_valid, self.Y_valid = self.getXYValidate(tag='annot3')

    def saveModel(self):
        self.fitRM()
        with open(self.model_dir+'RMannot3model.pkl', 'wb') as f:
            pickle.dump(self.RMmodel, f)


class BuildAnnot2Models(BuildClassicalModel):
    def __init__(self):
        super(BuildAnnot2Models, self).__init__()
        self.X_train, self.Y_train = self.getXYTrain(tag='annot2')
        self.X_valid, self.Y_valid = self.getXYValidate(tag='annot2')

    def saveModel(self):
        self.fitRM()
        with open(self.model_dir+'RMannot2model.pkl', 'wb') as f:
            pickle.dump(self.RMmodel, f)


class BuildAnnot1Models(BuildClassicalModel):
    def __init__(self):
        super(BuildAnnot1Models, self).__init__()
        self.X_train, self.Y_train = self.getXYTrain(tag='annot1')
        self.X_valid, self.Y_valid = self.getXYValidate(tag='annot1')

    def saveModel(self):
        self.fitRM()
        with open(self.model_dir+'RMannot1model.pkl', 'wb') as f:
            pickle.dump(self.RMmodel, f)
