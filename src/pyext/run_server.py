from model import EnsembleClassifier,PathwayClassifier,PathwaySimilarity
from gensim.models.fasttext import FastText as FT_gensim


class RunServerClassifier(object):
	'''

	'''
	def __init__(self):
		self.path='../../data/'
		self.models='../../data/models/'
		self.results='../../data/results/'
		self.fname='df_metacyc_multilabel_1.pkl'
		self.kegg='df_kegg.pkl'
		self.model_name='model5-10-100.model'
		self.model=FT_gensim.load(self.models+'model5-10-100.model')
		self.classifier=self.results+'enspickle_model.pkl'


	def Check_format(self,sample):
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
			if len(list(ec_class))>=7: 
				printed_text= "Input not in the right format, you are not using EC classe 1-7"
				return 1,printed_text           
		except:
				printed_text= "Input not in the right format, you might have forgotten using ec or the semicolon between ec and the numbers"
				return 1,printed_text

		try:
			# check if ec class digits are int and are less than 7
			ec_class_ind=[i.split(':')[1].split('.') for i in sample_list]
			ec_class=set([j for i in ec_class_ind for j in i ])
			ec_class_incorrect= [i for i in list(ec_class) if int(i) not in range(1,8)]
			if len(ec_class_incorrect)>0: 
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

	def run_classifier(self,):
		with open(pkl_filename, 'rb') as file:
			pickle_model = pickle.load(file)
		PW=PathwayClassifier()
		data_df=PW.clean_dataframe()
		model_gensim=PW.model
		X_train,Y_train,X_test,Y_test=PW.get_stratified_categories(data_df,model_gensim)
		trained_model=PW.model_validation(X_train,Y_train,X_test,Y_test)
		return trained_model

if __name__=='__main__':
	R=RunServerClassifier()
	print (R.Check_format('ec:1.1.1,ec:1.1,ec:2.4.5.1,ec:1.1.1.1'))

