import pickle
import pandas as pd
import numpy as np

class PathwayDF(object):
	def __init__(self):
		self.path='../../data/'

	def pathway_dict(self,file='pathway-links.dat'):
		pathway_names={}
		sf=open(self.path+file)
		for i in sf.readlines():
			lines=i.strip().split()
			pathway_names[lines[0]]=' '.join(lines[1:])
		return pathway_names

	def rxn_dict(self,file='reaction-links.dat'):
		rxn_ec={};rxn_no_ec=[]
		sf=open(self.path+file)
		for i in sf.readlines():
			lines=i.strip().split()
			if len(lines)>1:
				rxn_ec[lines[0]]=(' '.join(lines[1:])).lower().replace('-',':')
			else:
				rxn_ec[lines[0]]=''
				rxn_no_ec.append(lines[0])
		return rxn_ec,rxn_no_ec

	def generate_all_lines(self,file='pathway_cropped.dat'):
		all_lines=[]
		sf=open(self.path+file, encoding='utf-8',
					 errors='ignore')
		for i,j in enumerate(sf.readlines()):
			lines=j.strip()
			all_lines.append(lines)        
		return all_lines

	def get_pwy_rxns(self):
		pathway_rxns={}
		all_lines=self.generate_all_lines()
		for i,j in enumerate(all_lines):
			if 'UNIQUE-ID' in j:
				pathway_rxns[j.split()[-1]]=[]
				for m,n in enumerate(all_lines[i+1:]):
					if 'UNIQUE-ID' not in n:
						cleaned_line=n.replace("(",'').replace(')','').replace("\"",'')
						pathway_rxns[j.split()[-1]]=pathway_rxns[j.split()[-1]]+[i for i in cleaned_line.split() if 'RXN' in i]
					if (i+1+m+1)<len(all_lines) and 'UNIQUE-ID' in all_lines[i+1+m+1]:
						break
		return pathway_rxns

	def get_pwy_rxnsandpwy(self):
		all_lines=self.generate_all_lines()
		pathway_rxnsandpwy={}
		for i,j in enumerate(all_lines):
			if 'UNIQUE-ID' in j:
				pathway_rxnsandpwy[j.split()[-1]]=[]
				for m,n in enumerate(all_lines[i+1:]):
					if 'UNIQUE-ID' not in n:
						cleaned_line=n.replace("(",'').replace(')','').replace("\"",'')
						pathway_rxnsandpwy[j.split()[-1]]=pathway_rxns[j.split()[-1]]+[i for i in cleaned_line.split() if ('RXN' in i) or ('PWY' in i)]
					if (i+1+m+1)<len(all_lines) and 'UNIQUE-ID' in all_lines[i+1+m+1]:
						break
		return pathway_rxnsandpwy

	def get_ec_for_pwy(self,pathway_rxns,rxn_ec):
		pathway_ec={}
		for i,j in pathway_rxns.items():
			pathway_ec[i]=[]
			for m,n in enumerate(j):
				if len(rxn_ec[n])>1:
					pathway_ec[i].append(rxn_ec[n])     
		return pathway_ec

	def get_ec_for_pwy_all(self,pathway_rxnsandpwy,rxn_ec,pathway_ec):
		pathway_ec_all={}
		for i,j in pathway_rxnsandpwy.items():
			pathway_ec_all[i]=[]
			for m,n in enumerate(j):    
				if 'RXN' in n and len(rxn_ec[n])>1:
					pathway_ec_all[i].append(rxn_ec[n])  
				if 'PWY' in n and len(pathway_ec[n])>1:
					pathway_ec_all[i]=pathway_ec_all[i]+pathway_ec[n]
		return pathway_ec_all

	def get_key_from_val(self,dict1,val):
		key=list(dict1.keys())[list(dict1.values()).index(val)]
		return key

	def get_classes_dict(self):
		classes = pickle.load( open(self.path+ "metacyc_classes_all.pkl", "rb" ) )
		classes_dict={}
		for i,j in classes.items():
			for m,n in enumerate(j):
				classes_dict[n]=i
		return classes_dict

	def get_classes_nos_dict(self):
		Pathways=['Activation-Inactivation-Interconversion',
			  'Bioluminescence',
			  'Biosynthesis',
			  'Degradation',
			  'Detoxification',
			  'Energy-Metabolism',
			  'Glycan-Pathways',
			  'Macromolecule-Modification',
				'Metabolic-Clusters',
				 'Super-Pathways']
		classes_nos={j:i for i,j in enumerate(Pathways)}
		return classes_nos

	def get_pathway_class_dicts(self,pathway_ec_all,classes_dict,classes_nos):
		pathway_class={};pathway_class_label={}
		for i,j in pathway_ec.items():
			if i in list(classes_dict.keys()):
				pathway_class[i]=classes_dict[i]
				pathway_class_label[i]=classes_nos[classes_dict[i]]
		return pathway_class,pathway_class_label

	def create_df(self,pathway_ec,pathway_names,pathway_class,pathway_class_label):
		lst=[]
		for key,val in pathway_class.items():
			if len(pathway_ec[key])>3:
				lst.append([key,pathway_names[key],pathway_ec[key],list(set(pathway_ec[key])),val,pathway_class_label[key]])
		df=pd.DataFrame(lst,columns=['Map','Name','EC','EC_set','Label Name','Label'])
		return df
		
	def create_df(self,pathway_ec,pathway_names,pathway_class,pathway_class_label):
		lst=[]
		for key,val in pathway_class.items():
			if len(pathway_ec[key])>3:
				lst.append([key,pathway_names[key],pathway_ec[key],list(set(pathway_ec[key])),val,pathway_class_label[key]])
		df=pd.DataFrame(lst,columns=['Map','Name','EC','EC_set','Label Name','Label'])
		return df

	def create_df_all_labels(self,pathway_ec,pathway_names,Act,Bio,Bsy,Deg,Det,Ene,Gly,Mac):
		lst=[]
		for key,val in Act.items():
			if key in list(pathway_ec.keys()) and len(pathway_ec[key])>=3:
				lst.append([key,pathway_names[key],pathway_ec[key],list(set(pathway_ec[key])),val,
						   Bio[key],Bsy[key],Deg[key],Det[key],Ene[key],Gly[key],Mac[key]])
				
		df=pd.DataFrame(lst,columns=['Map','Name','EC','EC_set','Activation','Bioluminescence',
									'Biosynthesis','Degradation','Detoxification','Energy','Glycan','Macromolecule'])
		return df

	def EC_split(self,ec_list):
		EC=[]
		for i in ec_list:
			j=i.split()
			for m in j:
				EC.append(m)
		return EC

	def pwy_set(self,pwy):
		return list(set(pwy))

	def get_classes_dict_all(self): 
		classes = pickle.load( open(self.path+ "metacyc_classes_all.pkl", "rb" ) )
		Act=self.get_classes_dict_ml('Activation-Inactivation-Interconversion',classes)
		Bio=self.get_classes_dict_ml('Bioluminescence',classes)
		Bsy=self.get_classes_dict_ml('Biosynthesis',classes)
		Deg=self.get_classes_dict_ml('Degradation',classes)
		Det=self.get_classes_dict_ml('Detoxification',classes)
		Ene=self.get_classes_dict_ml('Energy-Metabolism',classes)
		Gly=self.get_classes_dict_ml('Glycan-Pathways',classes)
		Mac=self.get_classes_dict_ml('Macromolecule-Modification',classes)
		Met=self.get_classes_dict_ml('Metabolic-Clusters',classes)
		Sup=self.get_classes_dict_ml('Super-Pathways',classes)
		return Act,Bio,Bsy,Deg,Det,Ene,Gly,Mac
		
	def get_classes_dict_ml(self,tag,classes):
		tag_o={};
		rel_list=classes[tag]
		for i,j in enumerate(list(set(pathway_names.keys()))):
			if j in rel_list:
				tag_o[j]=1
			else:
				tag_o[j]=0
		return tag_o

if __name__=='__main__':
	path='../../data/'
	pathway_names=PathwayDF().pathway_dict()
	rxn_ec,rxn_no_ec=PathwayDF().rxn_dict()
	pathway_rxns=PathwayDF().get_pwy_rxns()
	pathway_ec=PathwayDF().get_ec_for_pwy(pathway_rxns,rxn_ec)
	Act,Bio,Bsy,Deg,Det,Ene,Gly,Mac=PathwayDF().get_classes_dict_all()
	metacyc_multilabel=PathwayDF().create_df_all_labels(pathway_ec,pathway_names,Act,Bio,Bsy,Deg,Det,Ene,Gly,Mac)
	pickle.dump(metacyc_multilabel, open(path+"df_metacyc_multilabel_1.pkl", "wb" ) )

	pathway_rxnsandpwy=PathwayDF().get_pwy_rxnsandpwy()
	pathway_ec_all=PathwayDF().get_ec_for_pwy_all(pathway_rxnsandpwy,rxn_ec,pathway_ec)
	metacyc_multilabel_all=PathwayDF().create_df_all_labels(pathway_ec_all,pathway_names,Act,Bio,Bsy,Deg,Det,Ene,Gly,Mac)
	pickle.dump(metacyc_multilabel_all, open(path+"df_metacyc_multilabel_2.pkl", "wb" ) )

