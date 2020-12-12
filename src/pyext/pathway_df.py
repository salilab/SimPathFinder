import pickle
import pandas as pd
import numpy as np
import os

class PathwayDF(object):
	def __init__(self):
		self.path='../../data/'

	def pathway_dict(self,data_dir,file='pathway-links.dat'):
		'''
		get names of all pathways from an organism folder 
		'''
		pathway_names={}
		sf=open(data_dir+file)
		for i in sf.readlines():
			lines=i.strip().split()
			pathway_names[lines[0]]=' '.join(lines[1:])
		return pathway_names

	def rxn_dict(self,data_dir,file='reaction-links.dat'):
		'''
		get set of reactions from organism folder 
		'''
		rxn_ec={};rxn_no_ec=[]
		sf=open(data_dir+file)
		for i in sf.readlines():
			lines=i.strip().split()
			if len(lines)>1:
				rxn_ec[lines[0]]=(' '.join(lines[1:])).lower().replace('-',':')
			else:
				rxn_ec[lines[0]]=''
				rxn_no_ec.append(lines[0])
		return rxn_ec,rxn_no_ec

	def generate_all_lines(self,data_dir,file='pathway_cropped.dat'):
		'''
		get all lines of pathways from organism folder
		'''
		all_lines=[]
		sf=open(data_dir+file, encoding='utf-8',
					 errors='ignore')
		for i,j in enumerate(sf.readlines()):
			lines=j.strip()
			all_lines.append(lines)        
		return all_lines

	def get_pwy_rxns(self,all_lines):
		'''
		create pathways with reactions by looking up rxn information from pathways
		'''
		pathway_rxns={}
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
		'''
		create pathways with reactions using predecessor information to get more context
		'''
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
		'''
		create a pathway representation with enzymes from reaction information 
		'''
		pathway_ec={}
		for i,j in pathway_rxns.items():
			pathway_ec[i]=[]
			for m,n in enumerate(j):
				if len(rxn_ec[n])>1:
					pathway_ec[i].append(rxn_ec[n])     
		return pathway_ec

	def get_ec_for_pwy_all(self,pathway_rxnsandpwy,rxn_ec,pathway_ec):
		'''
		create a pathway representation with enzymes from reaction information and prior pathway information
		'''
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
		'''
		get key from val
		'''
		key=list(dict1.keys())[list(dict1.values()).index(val)]
		return key

	def get_classes_dict(self):
		'''
		get pathway classes from metacyc [data obtained from scraping webpage: https://metacyc.org/META/class-tree?object=Pathways]
		'''
		classes = pickle.load( open(self.path+ "metacyc_classes_all.pkl", "rb" ) )
		classes_dict={}
		for i,j in classes.items():
			for m,n in enumerate(j):
				classes_dict[n]=i
		return classes_dict

	def get_classes_nos_dict(self):
		'''
		create labels for all classes for multi-label classification
		'''
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

	def create_df(self,pathway_ec,pathway_names,pathway_class,pathway_class_label):
		'''
		create df object for all pathways 
		'''
		lst=[]
		for key,val in pathway_class.items():
			if len(pathway_ec[key])>3:
				lst.append([key,pathway_names[key],pathway_ec[key],list(set(pathway_ec[key])),val,pathway_class_label[key]])
		df=pd.DataFrame(lst,columns=['Map','Name','EC','EC_set','Label Name','Label'])
		return df
		
	def create_df_all_labels(self,pathway_ec,pathway_names,Act,Bio,Bsy,Deg,Det,Ene,Gly,Mac):
		'''
		create df object for claases of all pathways 
		'''
		lst=[]
		for key,val in Act.items():
			if key in list(pathway_ec.keys()) and len(pathway_ec[key])>=3:
				lst.append([key,pathway_names[key],pathway_ec[key],list(set(pathway_ec[key])),val,
						   Bio[key],Bsy[key],Deg[key],Det[key],Ene[key],Gly[key],Mac[key]])
				
		df=pd.DataFrame(lst,columns=['Map','Name','EC','EC_set','Activation','Bioluminescence',
									'Biosynthesis','Degradation','Detoxification','Energy','Glycan','Macromolecule'])
		return df

	def pwy_set(self,pwy):
		return list(set(pwy))

	def get_classes_dict_all(self): 
		'''		classify pathways based on labels

		'''
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
		'''
		add binary labels to pathways based on class
		'''
		tag_o={};
		rel_list=classes[tag]
		for i,j in enumerate(list(set(self.pathway_names.keys()))):
			if j in rel_list:
				tag_o[j]=1
			else:
				tag_o[j]=0
		return tag_o

	def modify_pathway_file(self,dirname):
		'''
		clean the metacyc input file to remove descriptive text
		'''
		data_dir=self.path+dirname+'/'
		file='pathways.dat'
		file_out='pathway_cropped.dat'
		sf=open(data_dir+file,'r', encoding="utf8", errors='ignore')
		new_file=open(data_dir+file_out,'w+')
		for i in sf.readlines():
			if 'PREDECESSORS' in i or 'REACTION-LIST' in i or 'UNIQUE-ID' in i:
				new_file.write(i)
		new_file.close()

	def write_df_pkl_files(self,dirname,output_dir):
		'''
		write dataframe obj with pathways and enzyme composition from individual species 
		'''
		data_dir=self.path+dirname+'/'
		self.pathway_names=self.pathway_dict(data_dir)
		rxn_ec,rxn_no_ec=self.rxn_dict(data_dir)
		all_lines=self.generate_all_lines(data_dir)
		self.pathway_rxns=self.get_pwy_rxns(all_lines)
		self.pathway_ec=self.get_ec_for_pwy(self.pathway_rxns,rxn_ec)
		PWY_df=pd.DataFrame(self.pathway_ec.items(),columns=['PWY','EC'])
		PWY_df['org']=PWY_df['PWY']+' '+dirname
		pickle.dump(PWY_df, open(self.path+output_dir[0]+'/'+'PWY_'+dirname+".pkl", "wb" ) )

	def write_class_pkl_files(self,dirname,output_dir):
		'''
		write df obj to class-label pathways from individual speciea 
		'''
		classes_dict=self.get_classes_dict()
		classes_nos=self.get_classes_nos_dict()
		Act,Bio,Bsy,Deg,Det,Ene,Gly,Mac=self.get_classes_dict_all()
		PWY_class=self.create_df_all_labels(self.pathway_ec,self.pathway_names,Act,Bio,Bsy,Deg,Det,Ene,Gly,Mac)
		PWY_class['org']=PWY_class['Map']+' '+dirname
		print (PWY_class)
		pickle.dump(PWY_class, open(self.path+output_dir[1]+'/'+'PWY_class'+dirname+".pkl", "wb" ) )

	def all_pathway_tiers(self,output_dir=['PWY_df','PWY_class']):
		'''
		create pathway df obj and pathway class df objects 
		'''
		for name in output_dir:
			try:
				os.mkdir(self.path+name)
				print("Directory " , name ,  " Created ") 
			except FileExistsError:
				print("Directory " , name ,  " already exists")

		for x in os.walk(self.path):
			all_words=x[0].split('/')
			if 'PWY' not in all_words[-1] and len(all_words[-1])>=1:
				try:
					self.modify_pathway_file(all_words[-1])
					self.write_df_pkl_files(all_words[-1],output_dir)
					self.write_class_pkl_files(all_words[-1],output_dir)

				except:
					pass

	def combine_dfs(self,output_dir='PWY_df'):
		'''
		create pathway df obj and pathway class df objects 
		'''
		lst=list()
 		all_dfs=[df[2] for df in os.walk(self.path+output_dir)]
    	for df in all_dfs:
        	pwy_df=pickle.load(open(self.path+output_dir+i,'rb'))
        	lst.append(df)
        final_df=pd.concat(lst)
		pickle.dump(final_df, open(self.path+output_dir+"/final.pkl", "wb" ) )

