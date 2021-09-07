
from model import EnsembleClassifier,PathwayClassifier,PathwaySimilarity

def run_complete_annotation():
	'''
	run classifier with all 4 EC number digits
	'''
	PW=PathwayClassifier()
	data_df=PW.clean_dataframe()
	model_gensim=PW.model
	X_train,Y_train,X_test,Y_test=PW.get_stratified_categories(data_df,model_gensim)
	trained_model=PW.model_validation(X_train,Y_train,X_test,Y_test)
	return trained_model

def run_partial_annotation_3():
	'''
	run classifier with all 3 EC number digits
	'''
	PW=PathwayClassifier()
	data_df=PW.clean_dataframe()
	model_gensim=PW.model
	X_train,Y_train,X_test,Y_test=PW.get_stratified_categories_annot(data_df,model_gensim,annot=3)
	trained_model=PW.model_validation(X_train,Y_train,X_test,Y_test,subtitle='annot3')
	return trained_model

def run_partial_annotation_2():
	'''
	run classifier with all 2 EC number digits
	'''
	PW=PathwayClassifier()
	data_df=PW.clean_dataframe()
	model_gensim=PW.model
	X_train,Y_train,X_test,Y_test=PW.get_stratified_categories_annot(data_df,model_gensim,annot=2)
	trained_model=PW.model_validation(X_train,Y_train,X_test,Y_test,subtitle='annot2')
	return trained_model

def run_partial_annotation_1():
	'''
	run classifier with all 1 EC number digits
	'''
	PW=PathwayClassifier()
	data_df=PW.clean_dataframe()
	model_gensim=PW.model
	X_train,Y_train,X_test,Y_test=PW.get_stratified_categories_annot(data_df,model_gensim,annot=1)
	trained_model=PW.model_validation(X_train,Y_train,X_test,Y_test,subtitle='annot1')
	return trained_model

def run_control_OHC():
	'''
	run classifier with one-hot-coding (0s and 1s)
	'''
	PW=PathwayClassifier()
	data_df=PW.clean_dataframe()
	model_gensim=PW.model
	X_train,Y_train,X_test,Y_test=PW.get_X_and_Y_control(data_df)
	trained_model=PW.model_validation(X_train,Y_train,X_test,Y_test,subtitle='OHC')
	return trained_model

def check_similarity_single():
	'''
	find all similar pathways for 1 input 
	'''
	PW=PathwaySimilarity()
	valid=['ec:2.3.1.9','ec:2.8.3.5', 'ec:2.3.3.10',
 			'ec:4.1.1.4','ec:1.1.1.30','ec:4.1.3.4']
	df_simM,df_simK,df_sim=PW.get_ranking_single(valid,metacyc=True,kegg=True,combined=True)
	#print (df_sim.sort_values(by=['Similarity']).tail(10))
	return df_simM,df_simK,df_sim

def check_similarity_list():
	'''
	find all similar pathways for multiple inputs
	'''
	PW=PathwaySimilarity()
	valid=[['ec:2.3.1.9','ec:2.8.3.5', 'ec:2.3.3.10',
 			'ec:4.1.1.4','ec:1.1.1.30','ec:4.1.3.4']]
	df_sim=PW.get_ranking_list(valid)
	return df_sim

if __name__=='__main__':
	run_complete_annotation()
