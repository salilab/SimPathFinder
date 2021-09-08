import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from gensim.models.fasttext import FastText as FT_gensim
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns


class ExtractKEGGData(object):
    def __init__(self, data_dir='labeldata/',
                 final_model='RMmodel.pkl',
                 model_dir='models/'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.final_model = pickle.load(
            open(self.model_dir + final_model, "rb"))
        self.categories = {'detoxification': 0,
                           'activation': 1,
                           'biosynthesis': 2,
                           'degradation': 3,
                           'energy': 4,
                           'glycan': 5,
                           'macromolecule': 6},

    def get_pathways(self, file='df_kegg.pkl'):
        self.pwy = pickle.load(open(self.data_dir + file, "rb"))
        return self.pwy

    def fetchModel(self, modelName='tierT12_10_300'):
        self.model = FT_gensim.load(self.model_dir + modelName+'.model')
        return self.model

    def pwyVector(self, pwy):
        return np.mean(self.model.wv[pwy], axis=0)

    def getY(self, tag='biosynthesis'):
        self.df = self.get_pathways()
        self.df['Name'] = self.df['Name'].str.lower()
        self.df = self.df[self.df['Name'].apply(lambda x: tag in x)]
        self.Y = self.df.shape[0]*[1]
        return self.df

    def getYnot(self, tag='biosynthesis'):
        self.df = self.get_pathways()
        self.df['Name'] = self.df['Name'].str.lower()
        self.df = self.df[self.df['Name'].apply(lambda x: tag not in x)]
        self.Y = self.df.shape[0]*[1]
        return self.df

    def getY2(self, tag='biosynthesis'):
        self.df = self.get_pathways()
        self.df['Label Name'] = self.df['Label Name'].str.lower()
        self.df = self.df[self.df['Label Name'].apply(lambda x: tag in x)]
        self.Y = self.df.shape[0]*[1]
        return self.df

    def getY2not(self, tag='biosynthesis'):
        self.df = self.get_pathways()
        self.df['Label Name'] = self.df['Label Name'].str.lower()
        self.df = self.df[self.df['Label Name'].apply(lambda x: tag not in x)]
        self.Y = self.df.shape[0]*[1]
        return self.df

    def getX(self, tag='EC'):
        self.df['pathway_vector'] = self.df[tag].apply(self.pwyVector)
        self.X = list(self.df.pathway_vector)

    def applyPartialAnnot(self, pwy, annotLevel=3):
        output_ec = []
        for ec in pwy:
            temp = ec.split('.')
            if annotLevel == 3:
                if len(temp) == 4:
                    output_ec.append('.'.join(ec.split('.')[:-1]))
                else:
                    output_ec.append(ec)
            elif annotLevel == 2:
                if len(temp) == 4:
                    output_ec.append('.'.join(ec.split('.')[:-2]))
                elif len(temp) == 3:
                    output_ec.append('.'.join(ec.split('.')[:-1]))
                else:
                    output_ec.append(ec)
            elif annotLevel == 1:
                if len(temp) == 4:
                    output_ec.append('.'.join(ec.split('.')[:-3]))
                elif len(temp) == 3:
                    output_ec.append('.'.join(ec.split('.')[:-2]))
                elif len(temp) == 2:
                    output_ec.append('.'.join(ec.split('.')[:-1]))
                else:
                    output_ec.append(ec)

        return output_ec

    def createAllPartialAnnot(self):
        for annotLevel in range(1, 4):
            temp = 'annot'+str(annotLevel)
            self.df[temp] = self.df.EC.apply(
                self.applyPartialAnnot, annotLevel=annotLevel)
        return self.df

    def getPreds(self, tag='biosynthesis'):
        self.Y_pred = self.final_model.predict(
            self.X)[:, self.categories[0][tag]]

    def multiConfusionMatrix(self, tagY='biosynthesis',
                             tagP='biosynthesis',
                             annot='EC',
                             target1='Name',
                             target2='Pos'):
        if target1 == 'Name' and target2 == 'Pos':
            self.getY(tag=tagY)
        elif target1 == 'Name' and target2 == 'Neg':
            self.getYnot(tag=tagY)
        elif target1 == 'Label Name' and target2 == 'Pos':
            self.getY2(tag=tagY)
        elif target1 == 'Label Name' and target2 == 'Neg':
            self.getY2not(tag=tagY)
        else:
            print(
                'Wrong selections, target1 should be Name or Label Name and target2 should be Pos or Neg')

        self.createAllPartialAnnot()
        self.getX(tag=annot)
        self.getPreds(tag=tagP)
        conf_mat = confusion_matrix(self.Y, self.Y_pred)
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.add_subplot(111)
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap="crest",
                    linewidths=.5, cbar=True, annot_kws={"size": 14},
                    center=0, yticklabels=False, xticklabels=False,
                    cbar_kws={'orientation': 'horizontal'})
        plt.ylabel('Actual', fontsize=0)
        plt.xlabel('Predicted', fontsize=0)
        plt.title(tagP, fontsize=16)
        sns.despine(bottom=True, left=True)
        # cbar = ax.collections[0].colorbar
        plt.show()


class ExtractKEGGDataControl(ExtractKEGGData):
    def __init__(self, *args, **kwargs):
        super(ExtractKEGGDataControl, self).__init__(
            final_model='LRCmodel.pkl')

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

    def getX(self, tag='EC'):
        self.df['pathway_vector'] = self.df[tag].apply(self.getControl)
        self.X = list(self.df.pathway_vector)
