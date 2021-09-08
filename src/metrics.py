from keras import backend as K
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
import kerastuner
from kerastuner.tuners import RandomSearch
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Activation
from keras.models import Model, Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import tensorflow as tf
from models import ModelVectors
import matplotlib.pyplot as plt
from gensim.models.fasttext import FastText as FT_gensim
import seaborn as sns
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import *


class Metrics(object):
    def __init__(self, data_dir='labeldata/',
                 model_dir='models/',
                 final_model='RMmodel.pkl', *args, **kwargs):
        super(Metrics, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.final_model = pickle.load(
            open(self.model_dir + final_model, "rb"))

    def rocCurvePlot(self, categories=[], Yvalid_array=[], prediction_prob=[]):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        average_roc_auc = dict()
        colors = list(sns.color_palette("Set2"))
        for i in range(len(categories)):
            fpr[i], tpr[i], _ = roc_curve(Yvalid_array[:, i],
                                          prediction_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(Yvalid_array.ravel(),
                                                  prediction_prob.ravel())
        average_roc_auc["micro"] = roc_auc_score(Yvalid_array,
                                                 prediction_prob,
                                                 average="micro")

        for i in range(len(categories)):
            plt.plot(fpr[i], tpr[i], '-', label='%s (AUC %0.2f)' % (categories[i], roc_auc[i]),
                     lw=4, color=colors[i])
            plt.plot([0, 1], [0, 1], 'k--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.style.use('bmh')
        ax.set_facecolor('white')
        ax.grid(True)
        plt.legend(loc='lower right', fontsize=14,
                   edgecolor='white', facecolor=(1, 1, 1, 0.1))
        ax.set_xticks(ax.get_xticks()[1:])
        plt.tight_layout()
        ax.tick_params(axis='both', which='major', labelsize=16)

    def prCurvePlot(self, categories, Yvalid_array=[], prediction_prob=[]):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        precision = dict()
        recall = dict()
        average_precision = dict()
        colors = list(sns.color_palette("Set2"))
        for i in range(len(categories)):
            precision[i], recall[i], _ = precision_recall_curve(Yvalid_array[:, i],
                                                                prediction_prob[:, i])
            average_precision[i] = average_precision_score(Yvalid_array[:, i],
                                                           prediction_prob[:, i])

        precision["micro"], recall["micro"], _ = precision_recall_curve(Yvalid_array.ravel(),
                                                                        prediction_prob.ravel())
        average_precision["micro"] = average_precision_score(Yvalid_array, prediction_prob,
                                                             average="micro")
        plt.plot(recall["micro"], precision["micro"], lw=3,
                 label='micro-average (AUC {0:0.2f})'
                 ''.format(average_precision["micro"]))
        for i in range(len(categories)):
            plt.plot(recall[i], precision[i], '-', lw=3, color=colors[i],
                     label=' {0} (area = {1:0.2f})'.format(categories[i],
                                                           average_precision[i],
                                                           ))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=18)
        plt.ylabel('Precision', fontsize=18)
        plt.style.use('bmh')
        ax.set_facecolor('white')
        ax.grid(True)
        plt.legend(loc='lower left', fontsize=14,
                   edgecolor='white', facecolor=(1, 1, 1, 0.1))
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xticks(ax.get_xticks()[1:])
        plt.tight_layout()
        plt.show()

    def multiConfusionMatrix(self, Y_test, Y_pred):
        '''
        1. change x-ticks to T/F
        2. change y-ticks to T?F
        '''
        all_mats = multilabel_confusion_matrix(Y_test, Y_pred)
        count = 0
        for conf_mat in all_mats:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax = sns.heatmap(conf_mat, annot=True, fmt='d', cmap="crest",
                             linewidths=.5, cbar=True, annot_kws={"size": 14},
                             vmin=0, vmax=500, center=0, yticklabels=False, xticklabels=False,
                             cbar_kws={'orientation': 'horizontal'})
            plt.ylabel('Actual', fontsize=0)
            plt.xlabel('Predicted', fontsize=0)
            plt.title(Y_test.columns[count], fontsize=16)
            sns.despine(bottom=True, left=True)
            cbar = ax.collections[0].colorbar
            cbar.set_ticks([0, 250, 500])
            cbar.set_ticklabels([0, 250, 500])
            plt.show()
            count = count+1
        return all_mats

    def printAccuracyScore(self, Y_test, Y_pred):
        return accuracy_score(Y_test, Y_pred)

    def printHammingLoss(self, Y_test, Y_pred):
        return hamming_loss(Y_test, Y_pred)

    def evaluateAllLabels(self, ConfusionMatrix, categories, printOutput=False):
        categoriesMetrics = defaultdict()

        def evaluate(Tp, Tn, Fp, Fn):
            recall = Tp/(Tp+Fn)
            precision = Tp/(Tp+Fp)
            f1 = 2*(precision*recall)/(precision+recall)
            if printOutput:
                print('Recall = {:0.2f}%.'.format(recall))
                print('F1 score = {:0.2f}%.'.format(f1))
                print('Precision score = {:0.2f}%.'.format(precision))
            return [recall, precision, f1]

        for i in range(multiConfusionMatrix.shape[0]):
            tn = multiConfusionMatrix[i][0, 0]
            fp = multiConfusionMatrix[i][0, 1]
            fn = multiConfusionMatrix[i][1, 0]
            tp = multiConfusionMatrix[i][1, 1]
            if printOutput:
                print('Model Performance %s' % categories[i])
            categoriesMetrics[i] = evaluate(tp, tn, fp, fn)

    def evaluateMultilabelMetrics(self, Y_test, Y_pred):
        acc = []
        prec = []
        recall = []
        for i, j in enumerate(Y_test.to_numpy()):
            inter = [a for b, a in enumerate(j) if (
                a == Y_pred[i][b]) and (a == 1)]
            union_1 = [a for b, a in enumerate(
                j) if a != Y_pred[i][b] and (a == 1)]
            union_2 = [a for b, a in enumerate(
                j) if a != Y_pred[i][b] and (Y_pred[i][b] == 1)]
            union_f = len(inter)+len(union_1)+len(union_2)
            if union_f > 0:
                acc.append(len(inter)/union_f)
            if ((len(union_1)+len(inter))) > 0:
                tempr = len(inter)/(len(union_1)+len(inter))
                recall.append(tempr)
            if ((len(union_2)+len(inter))) > 0:
                prec.append(len(inter)/(len(union_2)+len(inter)))
        return sum(acc)/len(acc), sum(recall)/len(acc), sum(prec)/len(acc), self.printAccuracyScore(Y_test, Y_pred), self.printHammingLoss(Y_test, Y_pred)


class EvaluateMetrics(Metrics, ModelVectors):
    def __init__(self, *args, **kwargs):
        super(EvaluateMetrics, self).__init__(*args, **kwargs)
        self.X_train, self.Y_train = self.getXYTrain()
        self.X_valid, self.Y_valid = self.getXYValidate()

    def rocValidate(self, X_valid=None, Y_valid=None):
        if not X_valid:
            X_valid = self.X_valid
        if not Y_valid:
            Y_valid = self.Y_valid
        Y_pred_prob = self.final_model.predict_proba(X_valid)
        Y_valid = Y_valid.to_numpy()
        self.rocCurvePlot(self.categories, Y_valid, Y_pred_prob)

    def prValidate(self, X_valid=None, Y_valid=None):
        if not X_valid:
            X_valid = self.X_valid
        if not Y_valid:
            Y_valid = self.Y_valid
        Y_pred_prob = self.final_model.predict_proba(X_valid)
        Y_valid = Y_valid.to_numpy()
        self.prCurvePlot(self.categories, Y_valid, Y_pred_prob)

    def confusionMatrix(self, X_valid=None, Y_valid=None):
        if not X_valid:
            X_valid = self.X_valid
        if not Y_valid:
            Y_valid = self.Y_valid
        Y_pred = self.final_model.predict(X_valid)
        self.multiConfusionMatrix(Y_valid, Y_pred)

    def confusionMatrixTrain(self, X_valid=None, Y_valid=None):
        if not X_valid:
            X_valid = self.X_train
        if not Y_valid:
            Y_valid = self.Y_train
        Y_pred = self.final_model.predict(X_valid)
        self.multiConfusionMatrix(Y_valid, Y_pred)

    def allMultilabelMetrics(self, X_valid=None, Y_valid=None):
        if not X_valid:
            X_valid = self.X_valid
        if not Y_valid:
            Y_valid = self.Y_valid
        Y_pred = self.final_model.predict(X_valid)
        return self.evaluateMultilabelMetrics(Y_valid, Y_pred)


class EvaluateControl(EvaluateMetrics):
    def __init__(self, *args, **kwargs):
        super(EvaluateControl, self).__init__(final_model='LRCmodel.pkl')
        self.X_train, self.Y_train, self.X_valid, self.Y_valid = self.getXYControl()


class EvaluateAnnot3(EvaluateMetrics):
    def __init__(self, *args, **kwargs):
        super(EvaluateAnnot3, self).__init__(final_model='RMmodel.pkl')
        self.X_train, self.Y_train = self.getXYTrain(tag='annot3')
        self.X_valid, self.Y_valid = self.getXYValidate(tag='annot3')


class EvaluateAnnot2(EvaluateMetrics):
    def __init__(self, *args, **kwargs):
        super(EvaluateAnnot2, self).__init__(final_model='RMmodel.pkl')
        self.X_train, self.Y_train = self.getXYTrain(tag='annot2')
        self.X_valid, self.Y_valid = self.getXYValidate(tag='annot2')


class EvaluateAnnot1(EvaluateMetrics):
    def __init__(self, *args, **kwargs):
        super(EvaluateAnnot1, self).__init__(final_model='RMmodel.pkl')
        self.X_train, self.Y_train = self.getXYTrain(tag='annot1')
        self.X_valid, self.Y_valid = self.getXYValidate(tag='annot1')


class combinedEvaluations(Metrics, ModelVectors):
    def __init__(self, final_model_name='RMmodel100.pkl',
                 control=True,
                 *args, **kwargs):
        self.final_model_name = final_model_name
        self.control = control
        super(combinedEvaluations, self).__init__(*args, **kwargs)
        self.X_valid, self.Y_valid = self.getXYValidate()
        self.Xa1_valid, self.Ya1_valid = self.getXYValidate(tag='annot1')
        self.Xa2_valid, self.Ya2_valid = self.getXYValidate(tag='annot2')
        self.Xa3_valid, self.Ya3_valid = self.getXYValidate(tag='annot3')
        self.XC_train, self.YC_train, self.XC_valid, self.YC_valid = self.getXYControl()
        self.original_model = pickle.load(
            open(self.model_dir + self.final_model_name, "rb"))

        self.Y_pred = self.original_model.predict(self.X_valid)
        self.Y_pred_prob = self.original_model.predict_proba(self.X_valid)

        self.Ya1_pred = self.original_model.predict(self.Xa1_valid)
        self.Ya1_pred_prob = self.original_model.predict_proba(self.Xa1_valid)

        self.Ya2_pred = self.original_model.predict(self.Xa2_valid)
        self.Ya2_pred_prob = self.original_model.predict_proba(self.Xa2_valid)

        self.Ya3_pred = self.original_model.predict(self.Xa3_valid)
        self.Ya3_pred_prob = self.original_model.predict_proba(self.Xa3_valid)
        if self.control:
            self.control_model = pickle.load(
                open(self.model_dir + 'LRCmodel.pkl', "rb"))
            self.YC_pred = self.control_model.predict(self.XC_valid)
            self.YC_pred_prob = self.control_model.predict_proba(self.XC_valid)

    def prCurvePlotAll(self):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        precision = dict()
        recall = dict()
        average_precision = dict()
        colors = list(sns.color_palette("husl"))
        Yvalid_array = self.Y_valid.to_numpy()
        if self.control:
            prediction_prob = [('Annot: 4', self.Y_pred_prob),
                               ('Annot: 3', self.Ya3_pred_prob),
                               ('Annot: 2', self.Ya2_pred_prob),
                               ('Annot: 1', self.Ya1_pred_prob),
                               ('OHC: ', self.YC_pred_prob)]
        else:
            prediction_prob = [('Annot: 4', self.Y_pred_prob),
                               ('Annot: 3', self.Ya3_pred_prob),
                               ('Annot: 2', self.Ya2_pred_prob),
                               ('Annot: 1', self.Ya1_pred_prob)]

        for i in range(len(prediction_prob)):
            precision["micro"], recall["micro"], _ = precision_recall_curve(Yvalid_array.ravel(),
                                                                            prediction_prob[i][1].ravel())
            average_precision["micro"] = average_precision_score(Yvalid_array, prediction_prob[i][1],
                                                                 average="micro")
            plt.plot(recall["micro"], precision["micro"], '--', lw=3, color=colors[i],
                     label=prediction_prob[i][0]+' (AUC {0:0.2f})'
                     ''.format(average_precision["micro"]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=18)
        plt.ylabel('Precision', fontsize=18)

        plt.style.use('bmh')
        ax.set_facecolor('white')
        ax.grid(True)
        plt.legend(loc='lower right', fontsize=14,
                   edgecolor='white', facecolor=(1, 1, 1, 0.1))
        ax.set_xticks(ax.get_xticks()[1:])
        plt.tight_layout()
        ax.tick_params(axis='both', which='major', labelsize=16)

    def rocCurvePlotAll(self):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        fpr = dict()
        tpr = dict()
        average_roc_auc = dict()
        colors = list(sns.color_palette("husl"))
        Yvalid_array = self.Y_valid.to_numpy()

        if self.control:
            prediction_prob = [('Annot: 4', self.Y_pred_prob),
                               ('Annot: 3', self.Ya3_pred_prob),
                               ('Annot: 2', self.Ya2_pred_prob),
                               ('Annot: 1', self.Ya1_pred_prob),
                               ('OHC: ', self.YC_pred_prob)]
        else:
            prediction_prob = [('Annot: 4', self.Y_pred_prob),
                               ('Annot: 3', self.Ya3_pred_prob),
                               ('Annot: 2', self.Ya2_pred_prob),
                               ('Annot: 1', self.Ya1_pred_prob)]

        for i in range(len(prediction_prob)):

            fpr["micro"], tpr["micro"], _ = roc_curve(Yvalid_array.ravel(),
                                                      prediction_prob[i][1].ravel())
            average_roc_auc["micro"] = roc_auc_score(Yvalid_array,
                                                     prediction_prob[i][1],
                                                     average="micro")

            plt.plot(fpr["micro"], tpr["micro"], '--', lw=3, color=colors[i],
                     label=prediction_prob[i][0]+' (AUC {0:0.2f})'
                     ''.format(average_roc_auc["micro"]))

            plt.plot([0, 1], [0, 1], 'k--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.style.use('bmh')
        ax.set_facecolor('white')
        ax.grid(True)
        plt.legend(loc='lower right', fontsize=14,
                   edgecolor='white', facecolor=(1, 1, 1, 0.1))
        ax.set_xticks(ax.get_xticks()[1:])
        plt.tight_layout()
        ax.tick_params(axis='both', which='major', labelsize=16)


class MetricsDNN(object):
    def __init__(self, data_dir='../labeldata/',
                 model_dir='../models/', *args, **kwargs):
        super(MetricsDNN, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.final_model = keras.models.load_model(
            '../dnnmodels//my_model_LSTMFTune.h5')
        print(self.final_model.summary())
        self.fetchModel()
        self.X_train = self.trainLSTM.EC.to_list()
        self.X_test = self.testLSTM.EC.to_list()
        self.X_valid = self.validateLSTM.EC.to_list()
        self.Y_train = self.trainLSTM[self.categories].values
        self.Y_valid = self.validateLSTM[self.categories].values
        self.Y_test = self.testLSTM[self.categories].values

    def rocCurvePlot(self, categories=[], Yvalid_array=[], prediction_prob=[]):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        average_roc_auc = dict()
        colors = list(sns.color_palette("Set2"))
        for i in range(len(categories)):
            fpr[i], tpr[i], _ = roc_curve(Yvalid_array[:, i],
                                          prediction_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(Yvalid_array.ravel(),
                                                  prediction_prob.ravel())
        average_roc_auc["micro"] = roc_auc_score(Yvalid_array,
                                                 prediction_prob,
                                                 average="micro")

        for i in range(len(categories)):
            plt.plot(fpr[i], tpr[i], '-', label='%s (AUC %0.2f)' % (categories[i], roc_auc[i]),
                     lw=4, color=colors[i])
            plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.style.use('bmh')
        ax.set_facecolor('white')
        ax.grid(True)
        plt.legend(loc='lower right', fontsize=16,
                   edgecolor='white', facecolor=(1, 1, 1, 0.1))
        ax.set_xticks(ax.get_xticks()[1:])
        plt.tight_layout()
        ax.tick_params(axis='both', which='major', labelsize=16)

    def prCurvePlot(self, categories, Yvalid_array=[], prediction_prob=[]):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        precision = dict()
        recall = dict()
        average_precision = dict()
        colors = list(sns.color_palette("Set2"))
        for i in range(len(categories)):
            precision[i], recall[i], _ = precision_recall_curve(Yvalid_array[:, i],
                                                                prediction_prob[:, i])
            average_precision[i] = average_precision_score(Yvalid_array[:, i],
                                                           prediction_prob[:, i])

        precision["micro"], recall["micro"], _ = precision_recall_curve(Yvalid_array.ravel(),
                                                                        prediction_prob.ravel())
        average_precision["micro"] = average_precision_score(Yvalid_array, prediction_prob,
                                                             average="micro")
        plt.plot(recall["micro"], precision["micro"], lw=3,
                 label='micro-average (area = {0:0.2f})'
                 ''.format(average_precision["micro"]))
        for i in range(len(categories)):
            plt.plot(recall[i], precision[i], '-', lw=3, color=colors[i],
                     label=' {0} (area = {1:0.2f})'.format(categories[i],
                                                           average_precision[i],
                                                           ))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=18)
        plt.ylabel('Precision', fontsize=18)
        plt.style.use('bmh')
        ax.set_facecolor('white')
        ax.grid(True)
        plt.legend(loc='lower right', fontsize=14,
                   edgecolor='white', facecolor=(1, 1, 1, 0.1))
        # plt.legend(loc='lower right',fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xticks(ax.get_xticks()[1:])
        plt.tight_layout()
        plt.show()

    def multiConfusionMatrix(self, Y_test, Y_pred):
        '''
        1. change x-ticks to T/F
        2. change y-ticks to T?F
        '''
        all_mats = multilabel_confusion_matrix(Y_test, Y_pred)
        count = 0
        for conf_mat in all_mats:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax = sns.heatmap(conf_mat, annot=True, fmt='d', cmap="crest",
                             linewidths=.5, cbar=True, annot_kws={"size": 16},
                             vmin=0, vmax=500, center=0, yticklabels=False, xticklabels=False,
                             cbar_kws={'orientation': 'horizontal'})
            plt.ylabel('Actual', fontsize=0)
            plt.xlabel('Predicted', fontsize=0)
            plt.title(self.categories[count], fontsize=16)
            sns.despine(bottom=True, left=True)
            cbar = ax.collections[0].colorbar
            cbar.set_ticks([0, 250, 500])
            cbar.set_ticklabels([0, 250, 500])
            plt.show()

            count = count+1
        return all_mats

    def printAccuracyScore(self, Y_test, Y_pred):
        return accuracy_score(Y_test, Y_pred)

    def printHammingLoss(self, Y_test, Y_pred):
        return hamming_loss(Y_test, Y_pred)

    def evaluateAllLabels(self, ConfusionMatrix, categories, printOutput=False):
        categoriesMetrics = defaultdict()

        def evaluate(Tp, Tn, Fp, Fn):
            recall = Tp/(Tp+Fn)
            precision = Tp/(Tp+Fp)
            f1 = 2*(precision*recall)/(precision+recall)
            if printOutput:
                print('Recall = {:0.2f}%.'.format(recall))
                print('F1 score = {:0.2f}%.'.format(f1))
                print('Precision score = {:0.2f}%.'.format(precision))
            return [recall, precision, f1]

        for i in range(multiConfusionMatrix.shape[0]):
            tn = multiConfusionMatrix[i][0, 0]
            fp = multiConfusionMatrix[i][0, 1]
            fn = multiConfusionMatrix[i][1, 0]
            tp = multiConfusionMatrix[i][1, 1]
            if printOutput:
                print('Model Performance %s' % categories[i])
            categoriesMetrics[i] = evaluate(tp, tn, fp, fn)

    def evaluateMultilabelMetrics(self, Y_test, Y_pred):
        acc = []
        prec = []
        recall = []
        for i, j in enumerate(Y_test):
            inter = [a for b, a in enumerate(j) if (
                a == Y_pred[i][b]) and (a == 1)]
            union_1 = [a for b, a in enumerate(
                j) if a != Y_pred[i][b] and (a == 1)]
            union_2 = [a for b, a in enumerate(
                j) if a != Y_pred[i][b] and (Y_pred[i][b] == 1)]
            union_f = len(inter)+len(union_1)+len(union_2)
            if union_f > 0:
                acc.append(len(inter)/union_f)
            if ((len(union_1)+len(inter))) > 0:
                tempr = len(inter)/(len(union_1)+len(inter))
                recall.append(tempr)
            if ((len(union_2)+len(inter))) > 0:
                prec.append(len(inter)/(len(union_2)+len(inter)))
        return sum(acc)/len(acc), sum(recall)/len(acc), sum(prec)/len(acc), self.printAccuracyScore(Y_test, Y_pred), self.printHammingLoss(Y_test, Y_pred)


class EvaluateMetricsDNN(MetricsDNN, ModelVectors):
    def __init__(self, *args, **kwargs):
        super(EvaluateMetricsDNN, self).__init__(*args, **kwargs)
        self.fetchModel()
        self.X_train = self.trainLSTM.EC.to_list()
        self.X_test = self.testLSTM.EC.to_list()
        self.X_valid = self.validateLSTM.EC.to_list()
        self.Y_train = self.trainLSTM[self.categories].values
        self.Y_valid = self.validateLSTM[self.categories].values
        self.Y_test = self.testLSTM[self.categories].values

    def cleanTextChar(self, num_words=10000,
                      oov_token='<OOV>',
                      padding='post',
                      train=[],
                      test=[],
                      valid=[]):
        if not train:
            train = [' '.join(i) for i in self.X_train]
        if not valid:
            valid = [' '.join(i) for i in self.X_valid]
        if not test:
            test = [' '.join(i) for i in self.X_test]

        self.token = Tokenizer(num_words=num_words,
                               oov_token=oov_token, char_level=True)
        self.token.fit_on_texts(train+valid+test)
        self.word_index = self.token.word_index
        seq_train = self.token.texts_to_sequences(train)
        self.paddedTrain = pad_sequences(seq_train, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_valid = self.token.texts_to_sequences(valid)
        self.paddedValid = pad_sequences(seq_valid, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_test = self.token.texts_to_sequences(test)
        self.paddedTest = pad_sequences(seq_test, maxlen=len(
            max(train, test, valid)), padding=padding)
        return self.paddedTrain, self.paddedTest, self.paddedValid, self.token, self.word_index

    def rocValidate(self, X_valid=None, Y_valid=None):
        if not X_valid:
            X_valid = self.paddedTrain
        if not Y_valid:
            Y_valid = self.Y_train
        Y_pred_prob = self.final_model.predict_proba(X_valid)
        Y_valid = Y_valid
        self.rocCurvePlot(self.categories, Y_valid, Y_pred_prob)

    def prValidate(self, X_valid=None, Y_valid=None):
        if not X_valid:
            X_valid = self.paddedValid
        if not Y_valid:
            Y_valid = self.Y_valid
        Y_pred_prob = self.final_model.predict_proba(X_valid)
        Y_valid = Y_valid
        self.prCurvePlot(self.categories, Y_valid, Y_pred_prob)

    def confusionMatrix(self, X_valid=None, Y_valid=None):
        if not X_valid:
            X_valid = self.paddedValid
        if not Y_valid:
            Y_valid = self.Y_valid
        Y_pred_prob = self.final_model.predict(X_valid)
        Y_pred = (Y_pred_prob > 0.5)*1
        self.multiConfusionMatrix(Y_valid, Y_pred)

    def confusionMatrixTrain(self, X_valid=None, Y_valid=None):
        if not X_valid:
            X_valid = self.paddedTrain
        if not Y_valid:
            Y_valid = self.Y_train
        Y_pred_prob = self.final_model.predict(X_valid)
        Y_pred = (Y_pred_prob > 0.5)*1
        self.multiConfusionMatrix(Y_valid, Y_pred)

    def allMultilabelMetrics(self, X_valid=None, Y_valid=None):
        if not X_valid:
            X_valid = self.paddedValid
        if not Y_valid:
            Y_valid = self.Y_valid
        Y_pred_prob = self.final_model.predict(X_valid)
        Y_pred = (Y_pred_prob > 0.5)*1
        return self.evaluateMultilabelMetrics(Y_valid, Y_pred)


class EvaluateMetricsAnnot3DNN(EvaluateMetricsDNN):
    def __init__(self, *args, **kwargs):
        super(EvaluateMetricsAnnot3DNN, self).__init__(*args, **kwargs)
        self.Xa3_valid = self.validateLSTM.annot3.to_list()
        self.Xa2_valid = self.validateLSTM.annot2.to_list()
        self.Xa1_valid = self.validateLSTM.annot1.to_list()

    def cleanTextChar(self, num_words=10000,
                      oov_token='<OOV>',
                      padding='post'):
        train = [' '.join(i) for i in self.X_train]
        valid = [' '.join(i) for i in self.X_valid]
        test = [' '.join(i) for i in self.X_test]
        valida3 = [' '.join(i) for i in self.Xa3_valid]
        valida2 = [' '.join(i) for i in self.Xa2_valid]
        valida1 = [' '.join(i) for i in self.Xa1_valid]
        self.token = Tokenizer(num_words=num_words,
                               oov_token=oov_token, char_level=True)
        self.token.fit_on_texts(train+valid+test)
        self.word_index = self.token.word_index
        seq_train = self.token.texts_to_sequences(train)
        self.paddedTrain = pad_sequences(seq_train, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_valid = self.token.texts_to_sequences(valid)
        self.paddedValid = pad_sequences(seq_valid, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_test = self.token.texts_to_sequences(test)
        self.paddedTest = pad_sequences(seq_test, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_valida3 = self.token.texts_to_sequences(valida3)
        self.paddedValida3 = pad_sequences(seq_valida3, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_valida2 = self.token.texts_to_sequences(valida2)
        self.paddedValida2 = pad_sequences(seq_valida2, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_valida1 = self.token.texts_to_sequences(valida1)
        self.paddedValida1 = pad_sequences(seq_valida1, maxlen=len(
            max(train, test, valid)), padding=padding)

        self.Y_pred_prob = self.final_model.predict_proba(self.paddedValid)
        self.Ya3_pred_prob = self.final_model.predict_proba(self.paddedValida3)
        self.Ya2_pred_prob = self.final_model.predict_proba(self.paddedValida2)
        self.Ya1_pred_prob = self.final_model.predict_proba(self.paddedValida1)

        return self.paddedValida3, self.paddedValida2, \
            self.paddedValida1, self.paddedValid, self.token, self.word_index

    def prCurvePlotAll(self):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        precision = dict()
        recall = dict()
        average_precision = dict()
        colors = list(sns.color_palette("husl"))
        Yvalid_array = self.Y_valid
        prediction_prob = [('Annot: 4', self.Y_pred_prob),
                           ('Annot: 3', self.Ya3_pred_prob),
                           ('Annot: 2', self.Ya2_pred_prob),
                           ('Annot: 1', self.Ya1_pred_prob)]

        for i in range(len(prediction_prob)):
            precision["micro"], recall["micro"], _ = precision_recall_curve(Yvalid_array.ravel(),
                                                                            prediction_prob[i][1].ravel())
            average_precision["micro"] = average_precision_score(Yvalid_array, prediction_prob[i][1],
                                                                 average="micro")
            plt.plot(recall["micro"], precision["micro"], '--', lw=3, color=colors[i],
                     label=prediction_prob[i][0]+' (AUC {0:0.2f})'
                     ''.format(average_precision["micro"]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=18)
        plt.ylabel('Precision', fontsize=18)
        plt.style.use('bmh')
        ax.set_facecolor('white')
        ax.grid(True)
        plt.legend(loc='lower left', fontsize=14,
                   edgecolor='white', facecolor=(1, 1, 1, 0.1))
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xticks(ax.get_xticks()[1:])
        plt.tight_layout()
        plt.show()

    def rocCurvePlotAll(self):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        fpr = dict()
        tpr = dict()
        average_roc_auc = dict()
        colors = list(sns.color_palette("husl"))
        Yvalid_array = self.Y_valid
        prediction_prob = [('Annot: 4', self.Y_pred_prob),
                           ('Annot: 3', self.Ya3_pred_prob),
                           ('Annot: 2', self.Ya2_pred_prob),
                           ('Annot: 1', self.Ya1_pred_prob)]

        for i in range(len(prediction_prob)):

            fpr["micro"], tpr["micro"], _ = roc_curve(Yvalid_array.ravel(),
                                                      prediction_prob[i][1].ravel())
            average_roc_auc["micro"] = roc_auc_score(Yvalid_array,
                                                     prediction_prob[i][1],
                                                     average="micro")

            plt.plot(fpr["micro"], tpr["micro"], '--', lw=3, color=colors[i],
                     label=prediction_prob[i][0]+' (AUC {0:0.2f})'
                     ''.format(average_roc_auc["micro"]))

            plt.plot([0, 1], [0, 1], 'k--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        # plt.title('Receiver operating characteristics',fontsize=18)
        plt.style.use('bmh')
        ax.set_facecolor('white')
        ax.grid(True)
        plt.legend(loc='lower right', fontsize=14,
                   edgecolor='white', facecolor=(1, 1, 1, 0.1))
        ax.set_xticks(ax.get_xticks()[1:])
        plt.tight_layout()
        ax.tick_params(axis='both', which='major', labelsize=16)


class EvaluateMetricsAnnotsDNN(EvaluateMetricsDNN):
    def __init__(self, *args, **kwargs):
        super(EvaluateMetricsAnnotsDNN, self).__init__(*args, **kwargs)
        self.Xa3_valid = self.validateLSTM.annot3.to_list()
        self.Xa2_valid = self.validateLSTM.annot2.to_list()
        self.Xa1_valid = self.validateLSTM.annot1.to_list()

    def cleanTextChar(self, num_words=10000,
                      oov_token='<OOV>',
                      padding='post'):
        train = [' '.join(i) for i in self.X_train]
        valid = [' '.join(i) for i in self.X_valid]
        test = [' '.join(i) for i in self.X_test]
        valida3 = [' '.join(i) for i in self.Xa3_valid]
        valida2 = [' '.join(i) for i in self.Xa2_valid]
        valida1 = [' '.join(i) for i in self.Xa1_valid]
        self.token = Tokenizer(num_words=num_words,
                               oov_token=oov_token, char_level=True)
        self.token.fit_on_texts(train+valid+test)
        self.word_index = self.token.word_index
        seq_train = self.token.texts_to_sequences(train)
        self.paddedTrain = pad_sequences(seq_train, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_valid = self.token.texts_to_sequences(valid)
        self.paddedValid = pad_sequences(seq_valid, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_test = self.token.texts_to_sequences(test)
        self.paddedTest = pad_sequences(seq_test, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_valida3 = self.token.texts_to_sequences(valida3)
        self.paddedValida3 = pad_sequences(seq_valida3, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_valida2 = self.token.texts_to_sequences(valida2)
        self.paddedValida2 = pad_sequences(seq_valida2, maxlen=len(
            max(train, test, valid)), padding=padding)
        seq_valida1 = self.token.texts_to_sequences(valida1)
        self.paddedValida1 = pad_sequences(seq_valida1, maxlen=len(
            max(train, test, valid)), padding=padding)

        self.Y_pred_prob = self.final_model.predict_proba(self.paddedValid)
        self.Ya3_pred_prob = self.final_model.predict_proba(self.paddedValida3)
        self.Ya2_pred_prob = self.final_model.predict_proba(self.paddedValida2)
        self.Ya1_pred_prob = self.final_model.predict_proba(self.paddedValida1)

        return self.paddedValida3, self.paddedValida2, \
            self.paddedValida1, self.paddedValid, self.token, self.word_index

    def allMultilabelMetrics(self, annot=4):
        if annot == 4:
            X_valid = self.paddedValid
        elif annot == 3:
            X_valid = self.paddedValida3
        elif annot == 2:
            X_valid = self.paddedValida2
        elif annot == 1:
            X_valid = self.paddedValida1
        Y_valid = self.Y_valid
        Y_pred_prob = self.final_model.predict(X_valid)
        Y_pred = (Y_pred_prob > 0.4)*1
        return self.evaluateMultilabelMetrics(Y_valid, Y_pred)
