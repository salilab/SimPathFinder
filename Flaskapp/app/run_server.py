from gensim.models.fasttext import FastText as FT_gensim
import pickle
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

class RunServerClassifier(object):
	'''

	'''
	def __init__(self):
		self.models='../../models/'
		self.data='../../labeldata/'
		self.metacyc='LabeledData.pkl'
		self.kegg='df_kegg.pkl'
		self.model=FT_gensim.load(self.models+'tierT12_5_3_5_300.model')
		self.classifier=self.models+'RMmodel.pkl'
		self.categories=['Detoxification','Activation',
						'Biosynthesis', 'Degradation', 'Energy', 'Glycan',
						'Macromolecule']


	def check_format(self,sample):
		sample_list=sample.lower().split(',')
		try:
			ec=[i.count('ec') for i in sample_list]

			if len(sample_list)!=sum(ec):
				printed_text="Input not in the right format, you might have forgotten using ec or the semicolon between ec and the numbers"
				return 1,printed_text
		except (SyntaxError, ValueError):
				printed_text= "Input not in the right format, you might have forgotten using ec or the semicolon between ec and the numbers"
				return 1,printed_text
		try:
			# check if correct number and type of ec classes are being used
			ec_class_ind=[i.split(':')[1].split('.') for i in sample_list]
			ec_class=set([j for i in ec_class_ind for j in i ])
			ec_class_isal=[j.isalpha() for i in ec_class for j in i]
			if True in ec_class_isal: 
				printed_text= "Input not in the right format, your ec number has strings"
				return 1,printed_text
		except (IndexError, SyntaxError, ValueError):
				printed_text= "Input not in the right format, you might have forgotten using ec or the semicolon between ec and the numbers"
				return 1,printed_text
		try:
			# check if correct number and type of ec classes are being used
			ec_class_1=set([int(i.split(':')[1].split('.')[0]) for i in sample_list])
			ec_class_2=[i for i in ec_class_1 if i >=7]
			if len(list(ec_class_1))>=7: 
				printed_text= "Input not in the right format, you are not using EC classe 1-7"
				return 1,printed_text   
			if len(ec_class_2)>0: 
				printed_text= "Input not in the right format, you are not using EC classe 1-7"
				return 1,printed_text           
        
		except (IndexError, SyntaxError, ValueError):
				printed_text= "Input not in the right format, you might have forgotten using ec or the semicolon between ec and the numbers"
				return 1,printed_text

		try:
			#check number of digits in ec numbers
			ec_nums=set([len(i.split(':')[1].split('.')) for i in sample_list])
			ec_nums_T=[True for i in ec_nums if i not in [2,3,4]]
			if len(ec_nums_T)>0:
				printed_text= "Input not in the right format, check your ec numbers"
				return 1,printed_text                
		except (IndexError, SyntaxError, ValueError):
				printed_text= "Input not in the right format, check your ec numbers"
				return 1,printed_text

		return 0,"Input in the right format"

	def get_pathway_vectors(self,sample_list):
		return np.mean(self.model.wv[sample_list],axis=0)

	def run_classifier(self,sample):
		sample_list=sample.lower().split(',')
		model_gensim=self.model
		X_test=self.get_pathway_vectors(sample_list).reshape(1,-1)
		with open(self.classifier, 'rb') as file:
			pickle_model = pickle.load(file)
		prediction_prob=pickle_model.predict_proba(X_test)
		Y_pred = pickle_model.predict(X_test)
		classes=[ i[0] for i in list(zip(self.categories,Y_pred[0])) if i[1]==1 ]
		classes_prob=[str(round(i[1],4)) for i in list(zip(self.categories,prediction_prob[0])) if i[0] in classes]
		output_text1= 'The predicted class/classes: ' +' ,'.join(classes) 
		output_text2='The predicted class probability/probabilities: ' +' ,'.join(classes_prob)
		return output_text1,output_text2

	def similarity_dict_metacyc(self):
		data_df_multi=pickle.load(open(self.data+self.metacyc,'rb'))
		data_Metacyc_part=data_df_multi[['Map','Name','EC']]
		EC_list=data_Metacyc_part['EC'].to_list()
		EC_dict={tup[0]:tup[1] for tup in list(zip(data_Metacyc_part['Map'],data_Metacyc_part['EC']))}
		EC_name={tup[0]:tup[1] for tup in list(zip(data_Metacyc_part['Map'],data_Metacyc_part['Name']))}
		return EC_list,EC_dict,EC_name

	def similarity_dict_kegg(self):
		data_df_multi=pickle.load(open(self.data+self.kegg,'rb'))
		data_Metacyc_part=data_df_multi[['Map','Name','EC']]
		EC_list=data_Metacyc_part['EC'].to_list()
		EC_dict={tup[0]:tup[1] for tup in list(zip(data_Metacyc_part['Map'],data_Metacyc_part['EC']))}
		EC_name={tup[0]:tup[1] for tup in list(zip(data_Metacyc_part['Map'],data_Metacyc_part['Name']))}
		return EC_list,EC_dict,EC_name

	def get_ranking_single(self,valid,metacyc=True,kegg=True):
		
		if metacyc:
			data,data_dict,data_dict_name=self.similarity_dict_metacyc()
			df_simM=pd.DataFrame(columns=['Pathway_test','Pathway_train','Similarity'])
			for k,l in enumerate(data):
				sim=self.model.wv.n_similarity(valid, data[k])
				pathway=list(data_dict.keys())[k]
				pathway_name=data_dict_name[list(data_dict.keys())[k]]
				perc=round(sim*100,2)
				if sim>0:
					lst=[['UNK',list(data_dict.keys())[k],pathway_name,sim,perc]]
					df_simM=df_simM.append(pd.DataFrame(lst,columns=['Pathway_test','Pathway_train','Pathway_name','Similarity','Percentage']))

		if kegg:
			data,data_dict,data_dict_name=self.similarity_dict_kegg()
			df_simK=pd.DataFrame(columns=['Pathway_test','Pathway_train','Similarity'])
			for k,l in enumerate(data):
				sim=self.model.wv.n_similarity(valid, data[k])
				pathway=list(data_dict.keys())[k]
				pathway_name=data_dict_name[list(data_dict.keys())[k]]
				perc=round(sim*100,2)
				if sim>0:
					lst=[['UNK',list(data_dict.keys())[k],pathway_name,sim,perc]]
					df_simK=df_simK.append(pd.DataFrame(lst,columns=['Pathway_test','Pathway_train','Pathway_name','Similarity','Percentage']))
		
		return df_simM,df_simK


	def run_similarity(self,sample):
		sample_list=sample.lower().split(',')
		df_simM,df_simK=self.get_ranking_single(valid=sample_list)
		df_simM['Pathway Link']=df_simM['Pathway_train'].apply(lambda x:'https://biocyc.org/META/new-image?object='+str(x))
		Metacyc=df_simM[['Pathway_train','Percentage','Pathway Link']].sort_values(by=['Percentage'],ascending=False).head().values.tolist()
		Metacyc.insert(0,['Pathway','Percentage Similarity(%)','Pathway Link'])
		df_simK['Pathway Link']=df_simK['Pathway_train'].apply(lambda x:'https://www.genome.jp/dbget-bin/www_bget?'+str(x))
		Kegg=df_simK[['Pathway_train','Percentage','Pathway Link']].sort_values(by=['Percentage'],ascending=False).head().values.tolist()
		Kegg.insert(0,['Pathway','Percentage Similarity(%)','Pathway Link'])
		return Metacyc,Kegg

if __name__=='__main__':
	R=RunServerClassifier()
	#print (R.check_format('EC:1.1.1.1'))
	a,b=R.run_similarity('ec:1.2.3.4,ec:2.3.4.44,ec:3.1.23.1')
	#a,b=R.run_classifier('ec:1.2.3.4,ec:2.3.4.44,ec:3.1.23.1')
	print (a)
	print (b)

