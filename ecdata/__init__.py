import pickle
import pandas as pd
import numpy as np
import os
import random

class ExtractUnlabeledData(object):
    
    def __init__(self,data_dir='../'):
        self.data_dir=data_dir
        
    def getPwyDictClean(self, local_dir,file='pathway-links.dat'):
        self.pathway_names={}
        sf=open(local_dir+file,encoding="utf8", errors='ignore')
        for i in sf.readlines():
            lines=i.strip().split()
            if lines[0] != '#':
                self.pathway_names[lines[0]]=' '.join(lines[1:])
        return self.pathway_names
    
    def getPwyDictRaw(self,local_dir, file='pathways.dat'):
        self.pathway_names={}
        sf=open(local_dir+file,encoding="utf8", errors='ignore')
        for i in sf.readlines():
            line=i.strip().split()
            if line and line[0]=='UNIQUE-ID':
                temp=line[2]
            elif line and line[0]=='COMMON-NAME':
                self.pathway_names[temp]=' '.join(line[2:])
        return self.pathway_names

    def getRxnDictClean(self, local_dir,file='reaction-links.dat'):
        self.rxn_ec={}
        sf=open(local_dir+file,encoding="utf8", errors='ignore')
        for i in sf.readlines():
            lines=i.strip().split()
            if len(lines)>1 and lines[0] !='#':
                self.rxn_ec[lines[0]]=(' '.join(lines[1:])).lower().replace('-',':')
        return self.rxn_ec

    def getRxnDictRaw(self,local_dir, file='reactions.dat'):
        self.rxn_ec={}
        sf=open(local_dir+file,encoding="utf8", errors='ignore')
        for i in sf.readlines():
            line=i.strip().split()
            if line and line[0]=='UNIQUE-ID':
                temp_rxn=line[2]
            elif line and line[0]=='EC-NUMBER':
                self.rxn_ec[temp_rxn]=str(line[2]).lower().replace('-',':')
        return self.rxn_ec 
    
    def getPwyRxnMap(self,local_dir, file='pathways.dat'):
        self.pathway_rxn={}
        sf=open(local_dir+file, encoding='utf-8', errors='ignore')
        for i,j in enumerate(sf.readlines()):
            lines=j.strip().split()
            if lines and lines[0] == 'UNIQUE-ID':
                self.pathway_rxn[lines[2]]=[]
                temp_id=lines[2]
            elif lines and lines[0]=='REACTION-LIST':
                self.pathway_rxn[temp_id].append(lines[2])            
        return self.pathway_rxn
    
    def getPwyRxnPredMap(self,local_dir, file='pathways.dat'):
        self.pathway_pred_rxn={}
        try:
            sf=open(local_dir+file, encoding='utf-8', errors='ignore')
            for i,j in enumerate(sf.readlines()):
                lines=j.strip().split()
                if lines and lines[0] == 'UNIQUE-ID':
                    self.pathway_pred_rxn[lines[2]]=[]
                    temp_id=lines[2]
                elif lines and lines[0]=='PREDECESSORS':
                    try:
                        self.pathway_pred_rxn[temp_id].append(lines[2].replace('(',''))
                    except IndexError:
                        pass
                    try:
                        self.pathway_pred_rxn[temp_id].append(lines[3].replace(')',''))
                    except IndexError:
                        pass
                elif lines and lines[0]=='REACTION-LIST':
                    self.pathway_pred_rxn[temp_id].append(lines[2])
        except:
            pass
        return self.pathway_pred_rxn
    
    def globalRxnECMap(self):
        alld=[x[0]+'/' for x in os.walk(self.data_dir)]
        self.rxn_ec_global=dict()
        for i in alld[1:]:
            if i not in ['../pyext/','../pyext/.ipynb_checkpoints/','../pyext/data']:
                try:
                    temp=self.getRxnDictClean(i)
                    self.rxn_ec_global.update(temp)
                except FileNotFoundError:
                    try:
                        temp=self.getRxnDictRaw(i)
                        self.rxn_ec_global.update(temp)
                    except:
                        pass
        return self.rxn_ec_global
    
    def globalPwyRxnMap(self):
        alld=[x[0]+'/' for x in os.walk(self.data_dir)]
        pwy_rxn_global=[]
        ecrxnmap=self.globalRxnECMap()
        for i in alld[1:]:
            if i not in ['../pyext/','../pyext/.ipynb_checkpoints/','../pyext/data',
                         '../pyext/ecdata/']:
                pwyrxn=self.getPwyRxnPredMap(i)
                temp=pd.DataFrame(pwyrxn.items())
                pwy_rxn_global.append(temp)
        self.pwy_rxn_all=pd.concat(pwy_rxn_global)
        return self.pwy_rxn_all

    def globalPwyRxnEnzDF(self):
        #self.pwy_rxn_all=self.globalPwyRxnMap()
        # print (vars(self).keys())
        self.pwy_rxn_all=self.pwy_rxn_all.rename(columns = {0:'PWY',1:'RXN'}, inplace = False)
        def mapRxntoEC(rxn):
            output=[]
            for i in rxn:
                self.rxn_ec_global.get(i,'None')
                output.append(self.rxn_ec_global.get(i))
            return output
        self.pwy_rxn_all['EC']=self.pwy_rxn_all['RXN'].apply(mapRxntoEC)
        return self.pwy_rxn_all

    def finalEC(self):  
        EC_all=self.pwy_rxn_all['EC'].to_list()
        self.EC_clean=[]
        for each in EC_all:
            temp=[enz for enz in each if enz !=None]
            if len(temp)>1:
                self.EC_clean.append(temp)
        return self.EC_clean
    
    def saveAsPkl(self,default=True):
        if default:
            default_list=self.EC_clean
        sublabel=str(len(default_list))
        with open('allEC_'+sublabel+'.pkl', 'wb') as f:
            pickle.dump(default_list, f)
            
class SampleUnlabeledData(ExtractUnlabeledData):
    def __init__(self):
        super().__init__(data_dir='../')
        
    def generateSamples(self,sample_nos=1000):
        Enz=ExtractUnlabeledData()
        rxnEcMap=Enz.globalRxnECMap()
        pwyRxnMap=Enz.globalPwyRxnMap()
        pwyRxnEnzDF=Enz.globalPwyRxnEnzDF()
        enzyme_list=Enz.finalEC()
        all_enzymes=[j for i in enzyme_list for j in i]
        all_lengths=[len(i) for i in enzyme_list]
        extra_pathways=list()
        for i in range(sample_nos):
            temp_length=random.choice(all_lengths)
            temp_enzymes=[random.choice(all_enzymes) for i in range(temp_length)]
            extra_pathways.append(temp_enzymes)
        
        final_output=enzyme_list+extra_pathways
        return final_output
    
    def saveSamples(self,sample_nos=1000):
        samples=self.generateSamples(sample_nos)
        sublabel=str(sample_nos)
        with open('sampledEC_'+sublabel+'.pkl', 'wb') as f:
            pickle.dump(samples, f)


if __name__ == "__main__":
    S=SampleUnlabeledData()
    for i in range(0,60000,10000):
        S.saveSamples(sample_nos=i)
