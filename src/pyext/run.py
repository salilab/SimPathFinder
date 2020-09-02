
from model import EnsembleClassifier,PathwayClassifier,PathwaySimilarity

def run_complete_annotation():
	PW=PathwayClassifier()
	data_df=PW.clean_dataframe()
	model_gensim=PW.model
	X_train,Y_train,X_test,Y_test=PW.get_80_cat(data_df,model_gensim)
	trained_model=PW.model_validation(X_train,Y_train,X_test,Y_test)

if __name__=='__main__':
	run_complete_annotation()