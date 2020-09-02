import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab as plot
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import seaborn as sns
from collections import Counter

import numpy as np
import pickle

import pandas as pd
pd.set_option('mode.chained_assignment', None)

from gensim.models.fasttext import FastText as FT_gensim
import gensim
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
from gensim import models

from sklearn.model_selection import train_test_split,cross_validate, cross_val_score,StratifiedKFold,KFold
from sklearn.metrics import fbeta_score,f1_score,hamming_loss,accuracy_score, confusion_matrix,f1_score,precision_score,log_loss,classification_report,mean_squared_error,make_scorer,roc_curve,auc,precision_recall_curve
from scipy.stats import entropy
from sklearn.metrics import multilabel_confusion_matrix
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,VotingClassifier
from sklearn.svm import LinearSVC,SVC
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score,roc_auc_score


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, classifiers=None):
		self.classifiers = classifiers
		
	def fit(self, X, y):
		for classifier in self.classifiers:
			classifier[0]=classifier[1].fit(X, y)

	def predict_proba(self, X):
		self.predictions_ = list()
		for classifier in self.classifiers:
			self.predictions_.append(classifier[1].predict_proba(X))
		return np.mean(self.predictions_, axis=0)
	
	def predict(self, X):
		self.predictions_ = list()
		for classifier in self.classifiers:
			self.predictions_.append(classifier[1].predict_proba(X))
		return np.mean(self.predictions_, axis=0)


class PathwayClassifier(object):
	def __init__(self):
		self.path='../../data/'
		self.models='../../data/models/'
		self.fig='../../data/images/'
		self.results='../../data/results/'
		self.categories=['Detoxification','Activation',
						'Biosynthesis', 'Degradation', 'Energy', 'Glycan',
						'Macromolecule']
		self.fname='df_metacyc_multilabel_1.pkl'
		self.kegg='df_kegg.pkl'
		self.model_name='model5-10-100.model'
		self.model=FT_gensim.load(self.models+'model5-10-100.model')


	def clean_dataframe(self):
		data_df_multi=pickle.load(open(self.path+self.fname,'rb'))
		#all_enzymes=set([j for i in data_df_multi.EC for j in i])
		#for i in categories:
		#	data_df_multi=data_df_multi[data_df_multi[i]==0]
		return data_df_multi

	def get_all_enzymes(self):
		df_metacyc=pickle.load(open(self.path+self.fname,'rb'))
		df_kegg=pickle.load(open(self.path+self.kegg,'rb'))
		all_enzymes=set([j for i in df_metacyc.EC for j in i]+[j for i in df_kegg.EC for j in i])
		return all_enzymes

	def get_EC_list(self,df):
		EC_list_init=df['EC'].values.tolist()
		EC_list=[list((i)) for i in EC_list_init]
		return EC_list

	def get_all_classes(self,df):
		classes=sorted(list(set(df['Label Name'].to_list())))
		return classes

	def get_train_and_valid(self,df):
		train,valid=train_test_split(df,test_size=0.20)
		return train,valid

	def partial_annotation3(self,pathway):
		ec=[]
		for i in pathway:
			temp=i.split('.')
			if len(temp)==4:
				ec.append('.'.join(i.split('.')[:-1]))
			else:
				ec.append(i)
		return ec

	def partial_annotation2(self,pathway):
		ec=[]
		for i in pathway:
			temp=i.split('.')
			if len(temp)==4:
				ec.append('.'.join(i.split('.')[:-2]))
			elif len(temp)==3:
				ec.append('.'.join(i.split('.')[:-1]))
			else:
				ec.append(i)
		return ec

	def partial_annotation1(self,pathway):
		ec=[]
		for i in pathway:
			temp=i.split('.')
			if len(temp)==4:
				ec.append('.'.join(i.split('.')[:-3]))
			elif len(temp)==3:
				ec.append('.'.join(i.split('.')[:-2]))
			elif len(temp)==2:
				ec.append('.'.join(i.split('.')[:-1]))
			else:
				ec.append(i)
		return ec

	def label_enzymes(self,ec_set):
		lst=[]
		all_enzymes=self.get_all_enzymes()
		for i in all_enzymes:
			if i in ec_set:
				lst.append(1)
			else:
				lst.append(0)
		return np.array(lst)

	def pathway_vector(self,pathway,model_gensim):
		return np.mean(model_gensim[pathway],axis=0)
	 
	def get_80_cat(self,df,model_gensim,categories=[]):
		if len(categories)<1:
			categories=self.categories
		df['pathway_vector'] = df.EC_set.apply(self.pathway_vector,model_gensim=model_gensim)
		for i,j in enumerate(categories):
			dft=df[df[j]==1]
			df1=dft.sample(int(0.8*dft.shape[0]))
			df2=dft[~dft['Map'].isin(df1.Map.values)]
			if i==0:
				train=df1;
				test=df2;
			else:
				train=pd.concat([train,df1]);
				test=pd.concat([test,df2]);
		X_train=list(train.pathway_vector)
		Y_train=train[categories]
		X_test=list(test.pathway_vector)
		Y_test=test[categories]
		return X_train,Y_train,X_test,Y_test

	def get_80_cat_annot(self,df,model_gensim,categories=[],annot=3):
		if len(categories)<1:
			categories=self.categories
		if annot==3:
			df['anot'] = df.EC_set.apply(self.partial_annotation3)
		elif annot==2: 
			df['anot'] = df.EC_set.apply(self.partial_annotation2)
		elif annot==1:
			df['anot'] = df.EC_set.apply(self.partial_annotation1)
		df['pathway_vector'] = df.anot.apply(self.pathway_vector,model_gensim=model_gensim)
		for i,j in enumerate(categories):
			dft=df[df[j]==1]
			df1=dft.sample(int(0.8*dft.shape[0]))
			df2=dft[~dft['Map'].isin(df1.Map.values)]
			#print (i,j,df1.shape,df2.shape)
			if i==0:
				train=df1;
				test=df2;
			else:
				train=pd.concat([train,df1]);
				test=pd.concat([test,df2]);
		X_train=list(train.pathway_vector)
		Y_train=train[categories]
		X_test=list(test.pathway_vector)
		Y_test=test[categories]
		return X_train,Y_train,X_test,Y_test

	def get_X_and_Y_control(self,data_df_multi,categories=[]):
		if len(categories)<1:
			categories=self.categories
		data_df_multi['control_label']=data_df_multi.EC_set.apply(self.label_enzymes)
		data_df,valid_df=self.get_train_and_valid(data_df_multi.copy())
		X_train=list(data_df.control_label)
		X_test=list(valid_df.control_label)
		Y_train=data_df[categories]
		Y_test=valid_df[categories]
		return X_train,Y_train,X_test,Y_test

	def strplot(self,df,title):
		fig = plt.figure(figsize=(10,7))
		ax = fig.add_subplot(111)
		ax=sns.boxplot(data=df)
		ax=sns.stripplot(data=df, 
				  size=8, jitter=True, edgecolor="gray", linewidth=2)
		#ax.set_ylim([0,1])
		ax.set_ylabel(title,fontsize=20)
		ax.set_xticklabels(df.columns)
		ax.tick_params(labelsize=16)
		fig.savefig(self.results+title+'.png', dpi=fig.dpi,bbox_inches='tight')
		plt.close()
		#ax.set_title(title,fontsize=16)
			
	def roc_curve_plot(self,test_array,prediction_prob,title,categories=[]):
		if len(categories)<1:
			categories=self.categories
		fig = plt.figure(figsize=(10,7))
		ax1 = fig.add_subplot(111)
		fpr=dict();tpr=dict();average_roc_auc = dict();roc_auc=dict()
		for i in range(len(categories)):
			fpr[i], tpr[i], _ = roc_curve(test_array[:, i], prediction_prob[:, i]) 
			roc_auc[i] = auc(fpr[i], tpr[i])
			average_roc_auc[i] = roc_auc_score(test_array[:, i],
															prediction_prob[:, i])	        
		fpr["micro"], tpr["micro"], _ = roc_curve(test_array.ravel(),
																	prediction_prob.ravel())
		
		average_roc_auc["micro"] = roc_auc_score(test_array, prediction_prob,
														 average="micro")

		plt.plot(fpr["micro"], tpr["micro"],lw=6,
			 label='micro-average ROC curve (area = {0:0.2f})'
				   ''.format(average_roc_auc["micro"]))
		for i in range(len(categories)):
			plt.plot(fpr[i], tpr[i],'--',lw=4,
				 label='ROC curve of class {0} (area = {1:0.2f})'
					   ''.format(categories[i], average_roc_auc[i]))
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate',fontsize=18)
		plt.ylabel('True Positive Rate',fontsize=18)
		plt.legend(bbox_to_anchor=(1.1, 1.05))
		ax1.tick_params(axis='both', which='major', labelsize=16)
		plt.title('Receiver operating characteristics',fontsize=18)   
		fig.savefig(self.results+title+'roc.png', dpi=fig.dpi,bbox_inches='tight')
		plt.close()

	def pr_curve_plot(self,test_array,prediction_prob,title,categories=[]):
		if len(categories)<1:
			categories=self.categories
		fig = plt.figure(figsize=(10,7))
		ax1 = fig.add_subplot(111)
		precision=dict();recall=dict();average_precision = dict()
		for i in range(len(categories)):
			precision[i], recall[i], _ = precision_recall_curve(test_array[:, i],
															prediction_prob[:, i])
			average_precision[i] = average_precision_score(test_array[:, i],
															prediction_prob[:, i])
		precision["micro"], recall["micro"], _ = precision_recall_curve(test_array.ravel(),
																		prediction_prob.ravel())
		average_precision["micro"] = average_precision_score(test_array, prediction_prob,
														 average="micro")
		plt.plot(recall["micro"], precision["micro"],lw=6,
			 label='micro-average Precision-recall curve (area = {0:0.2f})'
				   ''.format(average_precision["micro"]))
		for i in range(len(categories)):
			plt.plot(recall[i], precision[i],'--',lw=4,
				 label='Precision-recall curve of class {0} (area = {1:0.2f})'
					   ''.format(categories[i], average_precision[i]))
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('Recall',fontsize=18)
		plt.ylabel('Precision',fontsize=18)
		plt.legend(bbox_to_anchor=(1.1, 1.05))
		plt.title('Precision vs Recall',fontsize=18)
		ax1.tick_params(axis='both', which='major', labelsize=16)
		fig.savefig(self.results+title+'pr.png', dpi=fig.dpi,bbox_inches='tight')
		plt.close()

	def multi_cm(self,Y_test,Y_pred,title):
		all_mats=multilabel_confusion_matrix(Y_test,Y_pred);
		count=0;
		for conf_mat in all_mats:
			fig, ax = plt.subplots(figsize=(5,5));
			sns.heatmap(conf_mat,annot=True, fmt='d',cmap="Blues",linewidths=.5,cbar=False,annot_kws={"size": 14});
			plt.ylabel('Actual',fontsize=16);
			plt.xlabel('Predicted',fontsize=16);
			ax.set_xticklabels([0,1],fontsize=16);
			ax.set_yticklabels([0,1],fontsize=16);
			plt.title(Y_test.columns[count],fontsize=16);
			fig.savefig(self.results+str(Y_test.columns[count])+title+'cm.png', dpi=fig.dpi,bbox_inches='tight')
			count=count+1;
			plt.close()
		return all_mats
		
	def evaluate_ml_metrics(self,Y_test,Y_pred):
		acc=[];prec=[];recall=[];f1=[]
		for i,j in enumerate(Y_test.to_numpy()):
			inter=[a for b,a in enumerate(j) if (a==Y_pred[i][b]) and (a==1)]
			union_1=[a for b,a in enumerate(j) if a!=Y_pred[i][b] and (a==1)]
			union_2=[a for b,a in enumerate(j) if a!=Y_pred[i][b] and (Y_pred[i][b]==1)]
			union_f=len(inter)+len(union_1)+len(union_2)
			if union_f>0:
				acc.append(len(inter)/union_f)
			if ((len(union_1)+len(inter)))>0:
				recall.append(len(inter)/(len(union_1)+len(inter)))
			if ((len(union_2)+len(inter)))>0:
				prec.append(len(inter)/(len(union_2)+len(inter)))
		return sum(acc)/len(acc),sum(recall)/len(acc),sum(prec)/len(acc)

	def spot_check(self,X_train,Y_train,n_splits=20):
		models = []
		models.append(('SVC', OneVsRestClassifier(SVC(probability=True)),
					('RM', OneVsRestClassifier(RandomForestClassifier(criterion='entropy'))),
					('GB',OneVsRestClassifier(GradientBoostingClassifier())),
					('MLP',OneVsRestClassifier(MLPClassifier())),
					('LR',OneVsRestClassifier(LogisticRegression()))))
		acc = [];loss=[];names = []
		scoring = {'hamming':make_scorer(hamming_loss),
			   'Accuracy': make_scorer(accuracy_score)}
		for i,tup in enumerate(models):
			name=tup[0]
			model=tup[1]
			kfold=KFold(n_splits=n_splits, random_state=1, shuffle=True)
			cv_results=cross_validate(model, X_train, Y_train, cv=kfold, scoring=scoring,return_train_score=True)
			acc.append(cv_results['test_Accuracy'])
			loss.append(cv_results['test_hamming'])
			names.append(name)
		return names,acc,loss

	def spot_check_plots(self,name,acc,loss):
		df_acc=pd.DataFrame(acc).T
		df_acc.rename(columns={0:names[0],1:names[1],2:names[2],3:names[3],4:names[4]},inplace=True)
		df_loss=pd.DataFrame(loss).T
		df_loss.rename(columns={0:names[0],1:names[1],2:names[2],3:names[3],4:names[4]},inplace=True)
		self.strplot(df_acc,title='Accuracy-Exact match')
		self.strplot(df_loss,title='Hamming loss')

	def grid_search_rf(self,X_train,Y_train):
		param_grid_rm ={
		'estimator__n_estimators':[1500,1800,2000] ,
		 'estimator__min_samples_split': [6,8,10,12,14],
		 'estimator__min_samples_leaf': [2, 4,6,8],
		 'estimator__max_features': ['sqrt','auto'],
		 'estimator__max_depth': [200,400,600,800,1000],
		 'estimator__criterion': ['gini'] }
		scoring = {'hamming':make_scorer(hamming_loss),
			   'Accuracy': make_scorer(accuracy_score)}
		rf = OneVsRestClassifier(RandomForestClassifier())
		grid_search = GridSearchCV(estimator = rf, param_grid = param_grid_rm, cv = 3, n_jobs = -1, verbose = 2,scoring=make_scorer(accuracy_score))
		grid_search.fit(X_train,Y_train)
		grid_search.best_params_
		joblib.dump(grid_search.best_params_, self.models+'best_rf.pkl', compress = 1) # Only best parameters
		return grid_search

	def grid_search_svc(self,X_train,Y_train):
		param_grid_svc ={
		"estimator__C": [1,2,4,8,10,12],
		"estimator__kernel": ["poly","rbf","linear","sigmoid"],
		"estimator__degree":[1, 2, 3, 4],
		"estimator__gamma":['scale', 'auto'],
		"estimator__class_weight":['balanced']}
		svc= OneVsRestClassifier(SVC(probability=True))
		grid_search_svc = GridSearchCV(estimator = svc, param_grid = param_grid_svc, cv = 5, n_jobs = -1, verbose = 2)
		grid_search_svc.fit(X_train,Y_train)
		grid_search_svc.best_params_
		joblib.dump(grid_search_svc.best_params_, self.models+'best_svc.pkl', compress = 1) # Only best parameters
		return grid_search_svc

	def fit_best_param(self,grid_search,X_train,Y_train, X_test, Y_test,title):
		best_rf=grid_search.best_estimator_
		model=best_rf.fit(X_train, Y_train)
		prediction=model.predict(X_test)
		prediction_prob=model.predict_proba(X_test)
		Y_pred = model.predict(X_test)
		test_array = Y_test.to_numpy()
		self.roc_curve_plot(categories,test_array,prediction_prob,title)
		self.pr_curve_plot(categories,test_array,prediction_prob,title)
		all_mats=self.multi_cm(Y_test,Y_pred,title)

	def run_best_model(self,categories=[],rf=True,svc=False):
		if len(categories)<1:
			categories=self.categories
		data_df_multi=self.clean_dataframe()
		X_train,Y_train,X_test,Y_test=self.get_80_cat(data_df_multi,categories)
		if rf:
			grid=self.grid_search_rf(X_train,Y_train)
			self.fit_best_param(grid,X_train,Y_train,X_test,Y_test,title='rf')
		if svc:
			grid=self.grid_search_svc(X_train,Y_train)
			self.fit_best_param(grid,X_train,Y_train,X_test,Y_test,title='svc')

	def model_validation(self,X_train,Y_train,X_test,Y_test,categories=[],subtitle='annot4'):
		if len(categories)<1:
			categories=self.categories
		models = []  
		models.append(('SVC', OneVsRestClassifier(SVC(probability=True,
												  C=8,
												  class_weight='balanced',
												  degree=1,
												  gamma='scale',
												  kernel='rbf',
												 break_ties=False, cache_size=200,
												 coef0=0.0,
												 decision_function_shape='ovr',
												  max_iter=-1,
												 random_state=None,
								  shrinking=True, tol=0.001, verbose=False))))
   
		models.append(('RM', OneVsRestClassifier(RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
						criterion='entropy', max_depth=None, max_features='auto',
						max_leaf_nodes=None, max_samples=None,
						min_impurity_decrease=0.0, min_impurity_split=None,
						min_samples_leaf=1, min_samples_split=2,
						min_weight_fraction_leaf=0.0, n_estimators=100,
						n_jobs=None, oob_score=False, random_state=None,
						verbose=0, warm_start=False))))
	
		models.append(('ens',OneVsRestClassifier(estimator=EnsembleClassifier(classifiers=[["SVC",models[0][1]],
																  ["RM",models[1][1]]]))))
		trained_models=[]
		for i,tup in enumerate(models):
			model=tup[1].fit(X_train, Y_train)
			trained_models.append(model)
			prediction_prob=model.predict_proba(X_test)
			Y_pred = model.predict(X_test)
			test_array = Y_test.to_numpy()
			title='valid'+'_'+str(tup[0])+'_'+subtitle
			self.roc_curve_plot(test_array,prediction_prob,title,categories)
			self.pr_curve_plot(test_array,prediction_prob,title,categories)
			all_mats=self.multi_cm(Y_test,Y_pred,title)
			acc=accuracy_score(Y_test, Y_pred),
			loss=hamming_loss(Y_test, Y_pred)
			acc_c,recall,prec=self.evaluate_ml_metrics(Y_test,Y_pred)
			print(acc,loss,acc_c,recall,prec, file=open(self.results+'_'+str(tup[0])+'_'+subtitle+"output.txt", "a"))
		return trained_models

	def kegg_analysis(self,model_gensim,classification_model,categories={'biosynthesis':3,'degradation':4,'glycan':6}):
	    kegg_df=pickle.load(open(self.path+self.kegg,'rb'))
	    kegg_df['Name']=kegg_df.Name.str.lower()
	    print (kegg_df.head())
	    print (kegg_df.shape)
	    predictions={}
	    for key,val in categories.items():
	        df=kegg_df[(kegg_df['Name'].str.contains(key))]
	        df['pathway_vector']=df.EC_set.apply(self.pathway_vector,model_gensim=model_gensim)
	        prediction=classification_model.predict(list(df.pathway_vector))
	        correct_p=[i for i,j in enumerate(prediction) if j[val-1]==1]
	        all_p=prediction.shape[0]
	        predictions[key]=(len(correct_p),all_p)
	    return predictions


class PathwaySimilarity(object):
	def __init__(self):
		self.path='../../data/'
		self.results='../../data/results/'
		self.fname='df_metacyc_multilabel_1.pkl'
		self.kegg='df_kegg.pkl'
		self.model_name='model5-10-100.model'
		self.models='../../data/models/'
		self.model=FT_gensim.load(self.models+'model5-10-100.model')

	def convert_text(self,Name):
		return Name.lower().replace('<i>','').replace('</i>','')

	def combined_df(self):
		data_df_multi=pickle.load(open(self.path+self.fname,'rb'))
		data_df=pickle.load(open(self.path+self.kegg,'rb'))
		data_KEGG_part=data_df[['Map','Name','EC','EC_set']]
		data_Metacyc_part=data_df_multi[['Map','Name','EC','EC_set']]
		Kegg_metacyc=pd.concat([data_KEGG_part,data_Metacyc_part])
		Kegg_metacyc['Length']=Kegg_metacyc['EC_set'].apply(lambda x: len(x))
		return Kegg_metacyc

	def similarity_dict(self):
		Kegg_metacyc=self.combined_df()
		Kegg_metacyc['Name']=Kegg_metacyc.Name.apply(self.convert_text)
		EC_list=Kegg_metacyc['EC_set'].to_list()
		EC_dict={tup[0]:tup[1] for tup in list(zip(Kegg_metacyc['Map'],Kegg_metacyc['EC_set']))}
		EC_name={tup[0]:tup[1] for tup in list(zip(Kegg_metacyc['Map'],Kegg_metacyc['Name']))}
		return EC_list,EC_dict,EC_name

	def get_ranking_list(self,valid):
		data,data_dict,data_dict_name=self.similarity_dict()
		valid_dict=dict()
		valid_dict['UNK']=valid[0]
		df_sim=pd.DataFrame(columns=['Pathway_test','Pathway_train','Similarity'])
		for i,j in enumerate(valid):
			for k,l in enumerate(data):
				sim=self.model.wv.n_similarity(valid[i], data[k])
				pathway=list(data_dict.keys())[k]
				pathway_name=data_dict_name[list(data_dict.keys())[k]]
				#jaccard=len(list(set(j).intersection(l)))/len(list(set(j).union(l)))
				perc=round(sim*100,2)
				if sim>0:
					lst=[[list(valid_dict.keys())[i],list(data_dict.keys())[k],pathway_name,sim,perc]]
					df_sim=df_sim.append(pd.DataFrame(lst,
												  columns=['Pathway_test','Pathway_train','Pathway_name','Similarity','Percentage']))
		return df_sim

	def get_ranking_single(self,valid):
		data,data_dict,data_dict_name=self.similarity_dict()
		df_sim=pd.DataFrame(columns=['Pathway_test','Pathway_train','Similarity'])
		for k,l in enumerate(data):
			sim=self.model.wv.n_similarity(valid, data[k])
			pathway=list(data_dict.keys())[k]
			pathway_name=data_dict_name[list(data_dict.keys())[k]]
				#jaccard=len(list(set(j).intersection(l)))/len(list(set(j).union(l)))
			perc=round(sim*100,2)
			if sim>0:
				lst=[['UNK',list(data_dict.keys())[k],pathway_name,sim,perc]]
				df_sim=df_sim.append(pd.DataFrame(lst,
												  columns=['Pathway_test','Pathway_train','Pathway_name','Similarity','Percentage']))
		return df_sim





