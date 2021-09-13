import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import random
from sklearn.model_selection import train_test_split


class ExtractUnlabeledData(object):

    def __init__(self, data_dir='../'):
        self.data_dir = data_dir
        # this data_dir refers to path to directories of different organisms
        # or biocyc Tier 1 and 2 organisms
        # pass local_dir to extract info

    def getPwyDictClean(self, local_dir: str, file='pathway-links.dat') -> dict:
        self.pathway_names = {}
        sf = open(local_dir+file, encoding="utf8", errors='ignore')
        for i in sf.readlines():
            lines = i.strip().split()
            if lines[0] != '#':
                self.pathway_names[lines[0]] = ' '.join(lines[1:])
        return self.pathway_names

    def getPwyDictRaw(self, local_dir: str, file='pathways.dat') -> dict:
        self.pathway_names = {}
        sf = open(local_dir+file, encoding="utf8", errors='ignore')
        for i in sf.readlines():
            line = i.strip().split()
            if line and line[0] == 'UNIQUE-ID':
                temp = line[2]
            elif line and line[0] == 'COMMON-NAME':
                self.pathway_names[temp] = ' '.join(line[2:])
        return self.pathway_names

    def getRxnDictClean(self, local_dir: str, file='reaction-links.dat') -> dict:
        self.rxn_ec = {}
        sf = open(local_dir+file, encoding="utf8", errors='ignore')
        for i in sf.readlines():
            lines = i.strip().split()
            if len(lines) > 1 and lines[0] != '#':
                self.rxn_ec[lines[0]] = (
                    ' '.join(lines[1:])).lower().replace('-', ':')
        return self.rxn_ec

    def getRxnDictRaw(self, local_dir: str, file='reactions.dat') -> dict:
        self.rxn_ec = {}
        sf = open(local_dir+file, encoding="utf8", errors='ignore')
        for i in sf.readlines():
            line = i.strip().split()
            if line and line[0] == 'UNIQUE-ID':
                temp_rxn = line[2]
            elif line and line[0] == 'EC-NUMBER':
                self.rxn_ec[temp_rxn] = str(line[2]).lower().replace('-', ':')
        return self.rxn_ec

    def getPwyRxnMap(self, local_dir: str, file='pathways.dat') -> dict:
        self.pathway_rxn = {}
        sf = open(local_dir+file, encoding='utf-8', errors='ignore')
        for i, j in enumerate(sf.readlines()):
            lines = j.strip().split()
            if lines and lines[0] == 'UNIQUE-ID':
                self.pathway_rxn[lines[2]] = []
                temp_id = lines[2]
            elif lines and lines[0] == 'REACTION-LIST':
                self.pathway_rxn[temp_id].append(lines[2])
        return self.pathway_rxn

    def getPwyRxnPredMap(self, local_dir: str, file='pathways.dat') -> dict:
        self.pathway_pred_rxn = {}
        sf = open(local_dir+file, encoding='utf-8', errors='ignore')
        for i, j in enumerate(sf.readlines()):
            lines = j.strip().split()
            if lines and lines[0] == 'UNIQUE-ID':
                self.pathway_pred_rxn[lines[2]] = []
                temp_id = lines[2]
            elif lines and lines[0] == 'PREDECESSORS':
                try:
                    self.pathway_pred_rxn[temp_id].append(
                        lines[2].replace('(', ''))
                except IndexError:
                    pass
                try:
                    self.pathway_pred_rxn[temp_id].append(
                        lines[3].replace(')', ''))
                except IndexError:
                    pass
            elif lines and lines[0] == 'REACTION-LIST':
                self.pathway_pred_rxn[temp_id].append(lines[2])
        return self.pathway_pred_rxn

    def globalRxnECMap(self) -> dict:
        alld = [x[0]+'/' for x in os.walk(self.data_dir)]
        self.rxn_ec_global = dict()
        for i in alld[1:]:
            if i not in ['../pyext/', '../pyext/.ipynb_checkpoints/', '../pyext/data']:
                try:
                    temp = self.getRxnDictClean(i)
                    self.rxn_ec_global.update(temp)
                except FileNotFoundError:
                    try:
                        temp = self.getRxnDictRaw(i)
                        self.rxn_ec_global.update(temp)
                    except IndexError:
                        pass
        return self.rxn_ec_global

    def globalPwyRxnMap(self) -> pd.DataFrame:
        alld = [x[0]+'/' for x in os.walk(self.data_dir)]
        pwy_rxn_global = []
        # ecrxnmap = self.globalRxnECMap()
        for i in alld[1:]:
            if i not in ['../pyext/', '../pyext/.ipynb_checkpoints/', '../pyext/data',
                         '../pyext/ecdata/']:
                pwyrxn = self.getPwyRxnPredMap(i)
                temp = pd.DataFrame(pwyrxn.items())
                pwy_rxn_global.append(temp)
        self.pwy_rxn_all = pd.concat(pwy_rxn_global)
        return self.pwy_rxn_all

    def globalPwyRxnEnzDF(self) -> pd.DataFrame:
        # self.pwy_rxn_all=self.globalPwyRxnMap()
        # print (vars(self).keys())
        self.pwy_rxn_all = self.pwy_rxn_all.rename(
            columns={0: 'PWY', 1: 'RXN'}, inplace=False)

        def mapRxntoEC(rxn):
            output = []
            for i in rxn:
                self.rxn_ec_global.get(i, 'None')
                output.append(self.rxn_ec_global.get(i))
            return output
        self.pwy_rxn_all['EC'] = self.pwy_rxn_all['RXN'].apply(mapRxntoEC)
        return self.pwy_rxn_all

    def finalEC(self) -> list:
        EC_all = self.pwy_rxn_all['EC'].to_list()
        self.EC_clean = []
        for each in EC_all:
            temp = [enz for enz in each if enz is not None]
            if len(temp) > 1:
                self.EC_clean.append(temp)
        return self.EC_clean

    def saveAsPkl(self, default=True):
        if default:
            default_list = self.EC_clean
        sublabel = str(len(default_list))
        with open('allEC_'+sublabel+'.pkl', 'wb') as f:
            pickle.dump(default_list, f)


class SampleUnlabeledData(ExtractUnlabeledData):
    def __init__(self):
        super().__init__(data_dir='../')

    def generateSamples(self, sample_nos=1000) -> list:
        Enz = ExtractUnlabeledData()
        # rxnEcMap = Enz.globalRxnECMap()
        # pwyRxnMap = Enz.globalPwyRxnMap()
        # pwyRxnEnzDF = Enz.globalPwyRxnEnzDF()
        enzyme_list = Enz.finalEC()
        all_enzymes = [j for i in enzyme_list for j in i]
        all_lengths = [len(i) for i in enzyme_list]
        extra_pathways = list()
        for i in range(sample_nos):
            temp_length = random.choice(all_lengths)
            temp_enzymes = [random.choice(all_enzymes)
                            for i in range(temp_length)]
            extra_pathways.append(temp_enzymes)

        final_output = enzyme_list+extra_pathways
        return final_output

    def saveSamples(self, sample_nos=1000):
        samples = self.generateSamples(sample_nos)
        sublabel = str(sample_nos)
        with open('sampledECT12_'+sublabel+'.pkl', 'wb') as f:
            pickle.dump(samples, f)


class ExtractLabeledData(object):
    def __init__(self, data_dir='../labeldata/'):
        self.data_dir = data_dir

    def get_pathways(self, file='MetacycPwyEC.pkl'):
        self.pwy = pickle.load(open(self.data_dir + file, "rb"))
        return self.pwy

    def get_pathway_names(self, file='MetacycPwyNames.pkl'):
        self.pwyNames = pickle.load(open(self.data_dir + file, "rb"))
        return self.pwyNames

    def get_classes_dict(self, file="metacyc_classes_all.pkl"):
        classes = pickle.load(open(self.data_dir + file, "rb"))
        self.Act = self.get_classes_dict_labels(
            'Activation-Inactivation-Interconversion', classes)
        self.Bio = self.get_classes_dict_labels('Bioluminescence', classes)
        self.Bsy = self.get_classes_dict_labels('Biosynthesis', classes)
        self.Deg = self.get_classes_dict_labels('Degradation', classes)
        self.Det = self.get_classes_dict_labels('Detoxification', classes)
        self.Ene = self.get_classes_dict_labels('Energy-Metabolism', classes)
        self.Gly = self.get_classes_dict_labels('Glycan-Pathways', classes)
        self.Mac = self.get_classes_dict_labels(
            'Macromolecule-Modification', classes)

    def get_classes_dict_labels(self, label: str, classes) -> dict:
        label_tags = {}
        label_list = classes[label]
        for _, pwy in enumerate(list(set(self.pwyNames.keys()))):
            if pwy in label_list:
                label_tags[pwy] = 1
            else:
                label_tags[pwy] = 0
        return label_tags

    def create_df_all_labels(self) -> df:
        lst = []
        for key, val in self.Act.items():
            if key in list(self.pwy.keys()) and len(self.pwy[key]) > 2:
                lst.append([key, self.pwyNames[key], self.pwy[key],
                            list(set(self.pwy[key])), val,
                            self.Bsy[key],
                            self.Deg[key], self.Det[key],
                            self.Ene[key], self.Gly[key],
                            self.Mac[key]])

        df = pd.DataFrame(lst, columns=['Map', 'Name', 'EC', 'EC_set', 'Activation',
                                        'Biosynthesis', 'Degradation', 'Detoxification', 'Energy', 'Glycan', 'Macromolecule'])

        with open(self.data_dir+'LabeledData.pkl', 'wb') as f:
            pickle.dump(df, f)
        return df


class BalanceLabelData(object):
    def __init__(self, data_dir='../labeldata/'):
        self.data_dir = data_dir

    def loadAllData(self, file='LabeledData.pkl') -> dict:
        self.allData = pickle.load(open(self.data_dir + file, "rb"))
        return self.allData

    def printStatsOnData(self):
        classes = self.allData.columns.to_list()[4:]
        for label in classes:
            temp = self.allData[self.allData[label] == 1]
            print(temp.shape[0], label)

    def applyPartialAnnot(self, pwy: list, annotLevel=3) -> list:
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

    def createAllPartialAnnot(self) -> pd.DataFrame:
        for annotLevel in range(1, 4):
            temp = 'annot'+str(annotLevel)
            self.allData[temp] = self.allData.EC.apply(
                self.applyPartialAnnot, annotLevel=annotLevel)
        return self.allData

    def splitData(self) -> (list, list):
        self.train, self.validate = \
            np.split(self.allData.sample(frac=1, random_state=42),
                     [int(.8*len(self.allData))])
        with open(self.data_dir+'TrainLabeledData.pkl', 'wb') as f:
            pickle.dump(self.train, f)
        with open(self.data_dir+'ValidateLabeledData.pkl', 'wb') as f:
            pickle.dump(self.validate, f)

        return self.train, self.validate

    def splitDataLSTM(self):
        self.train, self.validate, self.test = \
            np.split(self.allData.sample(frac=1, random_state=42),
                     [int(.6*len(self.allData)), int(.8*len(self.allData))])
        with open(self.data_dir+'ValidateLabeledDataLSTM.pkl', 'wb') as f:
            pickle.dump(self.validate, f)
        with open(self.data_dir+'TrainLabeledDataLSTM.pkl', 'wb') as f:
            pickle.dump(self.train, f)
        with open(self.data_dir+'testLabeledDataLSTM.pkl', 'wb') as f:
            pickle.dump(self.test, f)
