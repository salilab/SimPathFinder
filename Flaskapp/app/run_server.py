from model import EnsembleClassifier,PathwayClassifier,PathwaySimilarity
from gensim.models.fasttext import FastText as FT_gensim
import pickle

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class RunServerClassifier(object):
	'''

	'''
	def __init__(self):
		self.path='../../data/'
		self.models='../../data/models/'
		self.results='../../data/results/'
		self.fname='df_metacyc_multilabel_1.pkl'
		self.kegg='df_kegg.pkl'
		self.model=FT_gensim.load(self.models+'model5-10-100.model')
		self.classifier=self.results+'enspickle_model.pkl'


	def check_format(self,sample):
		sample_lower=sample.lower()
		try:
			sample_list=sample_lower.split(',')
		except :
			printed_text="Input not in the right format, you might have forgotten commas"
			return 1,printed_text
		try:
			ec=[i.count('ec') for i in sample_list]
			#print (ec)
			#print (sample_list)
			if len(sample_list)!=sum(ec):
				printed_text="Input not in the right format, you might have forgotten using ec or the semicolon between ec and the numbers"
				return 1,printed_text
		except:
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
		except:
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
        
		except:
				printed_text= "Input not in the right format, you might have forgotten using ec or the semicolon between ec and the numbers"
				return 1,printed_text

		try:
			#check number of digits in ec numbers
			ec_nums=set([len(i.split(':')[1].split('.')) for i in sample_list])
			ec_nums_T=[True for i in ec_nums if i not in [2,3,4]]
			#print ("ec_nums",ec_nums,ec_nums_T)
			if len(ec_nums_T)>0:
				printed_text= "Input not in the right format, check your ec numbers"
				return 1,printed_text                
		except:
				printed_text= "Input not in the right format, check your ec numbers"
				return 1,printed_text

		return 0,"Input in the right format"

	def run_classifier(self,sample):
		sample_list=sample.lower().split(',')
		PW=PathwayClassifier()
		model_gensim=self.model
		X_test=PW.pathway_vector(sample_list,self.model).reshape(1,-1)
		with open(self.classifier, 'rb') as file:
			pickle_model = pickle.load(file)

		prediction_prob=pickle_model.predict_proba(X_test)
		Y_pred = pickle_model.predict(X_test)
		classes=[ i[0] for i in list(zip(PW.categories,Y_pred[0])) if i[1]==1 ]
		classes_prob=[str(round(i[1],4)) for i in list(zip(PW.categories,prediction_prob[0])) if i[0] in classes]
		output_text1= 'The predicted class/classes: ' +' ,'.join(classes) 
		output_text2='The predicted class probability/probabilities: ' +' ,'.join(classes_prob)
		return output_text1,output_text2

	def run_similarity(self,sample):
		sample_list=sample.lower().split(',')
		PW=PathwaySimilarity()
		df_simM,df_simK,df_sim=PW.get_ranking_single(sample_list)
		df_simM['Pathway Link']=df_simM['Pathway_train'].apply(lambda x:'https://biocyc.org/META/new-image?object='+str(x))
		Metacyc=df_simM[['Pathway_train','Percentage','Pathway Link']].sort_values(by=['Percentage'],ascending=False).head().values.tolist()
		Metacyc.insert(0,['Pathway','Percentage Similarity(%)','Pathway Link'])
		df_simK['Pathway Link']=df_simK['Pathway_train'].apply(lambda x:'https://www.genome.jp/dbget-bin/www_bget?'+str(x))
		Kegg=df_simK[['Pathway_train','Percentage','Pathway Link']].sort_values(by=['Percentage'],ascending=False).head().values.tolist()
		Kegg.insert(0,['Pathway','Percentage Similarity(%)','Pathway Link'])
		return Metacyc,Kegg

if __name__=='__main__':
	R=RunServerClassifier()
	#print (R.check_format('ec:1.2.3.4,ec:2.3.4.44,ec:3.1.23.1'))
	R.run_similarity('ec:1.2.3.4,ec:2.3.4.44,ec:3.1.23.1')

