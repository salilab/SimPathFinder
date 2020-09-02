import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os,pickle
from collections import Counter
from gensim import corpora, models
from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath
from gensim.models.phrases import Phrases, Phraser
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import umap
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
sns.set(style='white', rc={'figure.figsize':(10,8)})
sns.set_style("whitegrid")
import itertools
import warnings
warnings.filterwarnings('ignore')

class Embeddings(object):
	def __init__(self):
		self.path='../../data/'
		self.models='../../data/models/'
		self.fig='../../data/images/'

	def get_EC(self):
		dataM=pickle.load(open(self.path+'df_metacyc_multilabel.pkl','rb'))
		dataK=pickle.load(open(self.path+'df_kegg.pkl','rb'))
		EC=dataM.EC.to_list() +dataK.EC.to_list()
		return EC

	def get_metacyc_df(self):
		dataM=pickle.load(open(self.path+'df_metacyc_multilabel.pkl','rb'))
		return dataM

	def FT(self,epoch,EC_lists,negative,min_n):
		model_gensim = FT_gensim(size=300)
		model_gensim.build_vocab(sentences=EC_lists,min_count=1)
		model_gensim.train(EC_lists, total_examples=len(EC_lists), epochs=epoch,min_n=min_n,window=5,sg=1,negative=negative)
		return model_gensim

	def create_embeddings(self,min_n,negative,epochs):
		combined_list = min_n, negative, epochs
		parameters=list(itertools.product(*combined_list))
		for i,j in enumerate(parameters):
			name='model'+str(j[0])+'-'+str(j[1])+'-'+str(j[2])
			model=self.FT(epoch=j[2],EC_lists=self.get_EC(),min_n=j[0],negative=j[1])
			model.save(Path+name+".model")

	def get_clusters(self,model_gensim,keys):
		embedding_clusters = []
		word_clusters=[]
		for word in keys:
			embeddings = []
			words = []
			for similar_word, _ in model_gensim.wv.most_similar(word, topn=100):
				words.append(similar_word)
				embeddings.append(model_gensim[similar_word])
			embedding_clusters.append(embeddings)
			word_clusters.append(words)
		return word_clusters,embedding_clusters

	def plot_umap(self,cluster_df,model_gensim,title_text):
		keys=['Degradation','Biosynthesis','Energy','Activation','Glycan','Macromolecule','Detoxification']
		fig = plt.figure(figsize=(10,7))
		ax1 = fig.add_subplot(111)
		for m,n in enumerate(keys):
			embeddings=[]
			labels=cluster_df[cluster_df[n]==1]#.value_counts()
			PWY=labels.EC_set.to_list()
			for l in PWY:
				embeddings.append(np.mean(model_gensim[l],axis=0))
			emb=np.array(embeddings)
			X_2d = umap.UMAP(random_state=42).fit_transform(emb)
			colors = 'r', 'g', 'b', 'orange', 'pink', 'm','purple','brown','y'
			ax1.scatter(X_2d[:, 0], X_2d[:, 1],c=colors[m],s=200,edgecolor='gray',label=keys[m])
		plt.xlabel('Dimension #1',size=16)
		plt.ylabel('Dimension #2',size=16)
		plt.grid(False)
		plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0.,fontsize=16)
		ax1.tick_params(axis='both', which='major', labelsize=0)
		plt.tight_layout() 
		fig.savefig(self.fig+title_text+'umap.png', dpi=fig.dpi,bbox_inches='tight')
	
	def plot_umap_kegg(self,cluster_df,model_gensim):
		keys=['Amino-acid-metabolism',
			'Biosynthesis-of-other-secondary-metabolites',
			'Carbohydrate-metabolism',
			'Energy-metabolism',
			'Glycan-biosynthesis-and-metabolism',
			'Lipid-metabolism',
			'Metabolism-of-cofactors-and-vitamins',
			'Metabolism-of-other-amino-acids',
			'Metabolism-of-terpenoids-and-polyketides',
			'Xenobiotics-biodegradation-and-metabolism']
		fig = plt.figure(figsize=(10,7))
		ax1 = fig.add_subplot(111)
		for m,n in enumerate(keys):
			embeddings=[]
			labels=cluster_df[cluster_df['Label Name']==n]#.value_counts()
			PWY=labels.EC_set.to_list()
			for l in PWY:
				embeddings.append(np.mean(model_gensim[l],axis=0))
			emb=np.array(embeddings)
			X_2d = umap.UMAP(random_state=42).fit_transform(emb)
			colors = 'r', 'g', 'b', 'orange', 'pink', 'm','purple','y','red','blue','green','black'
			ax1.scatter(X_2d[:, 0], X_2d[:, 1],c=colors[m],s=200,edgecolor='gray',label=keys[m])
		plt.xlabel('Dimension #1',size=16)
		plt.ylabel('Dimension #2',size=16)
		plt.grid(False)
		plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0.,fontsize=12)
		ax1.tick_params(axis='both', which='major', labelsize=0)
		plt.tight_layout() 
		fig.savefig(self.fig+'kegg.png', dpi=fig.dpi,bbox_inches='tight')

	def plot_umap_similar_words(self, labels, embedding_clusters, word_clusters,title_text):
		fig = plt.figure(figsize=(10, 7))
		ax = plt.subplot(111)
		colors = 'r', 'g', 'b', 'orange', 'pink', 'm','purple','brown'
		for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
			x = embeddings[:, 0]
			y = embeddings[:, 1]
			plt.scatter(x, y, c=[color], label=label,s=300,edgecolor='gray',);
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.xlabel('Dimension #1',size=16)
		plt.ylabel('Dimension #2',size=16)
		plt.grid(False)
		plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0.,fontsize=16)
		ax.tick_params(axis='both', which='major', labelsize=0)
		plt.tight_layout() 
		fig.savefig(self.fig+title_text+'.png', dpi=fig.dpi,bbox_inches='tight')

	def FT_ec_umap(self, model_gensim,title_text):
		keys=['ec:1','ec:2','ec:3','ec:4','ec:5','ec:6']
		word,embeddings=self.get_clusters(model_gensim,keys)
		embedding_clusters = np.array(embeddings)
		n, m, k = embedding_clusters.shape
		X_2d = np.array(umap.UMAP(random_state=42).fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n,m,2)
		self.plot_umap_similar_words(keys,X_2d,word,title_text+'ecumap')
	
	def FT_ec_tsne(self,model_gensim,title_text):
		keys=['ec:1','ec:2','ec:3','ec:4','ec:5','ec:6']
		word,embeddings=self.get_clusters(model_gensim,keys)
		embedding_clusters = np.array(embeddings)
		n, m, k = embedding_clusters.shape
		tsne_model_en_2d = TSNE(perplexity=25, n_components=2, init='pca', n_iter=5000, random_state=32)
		X_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
		self.plot_umap_similar_words(keys,X_2d,word,title_text+'ectsne')

	def plot_all_figures(self,epochs=[100,200,300],negative=[10,100,1000],min_n=[3]):
		#min_n=[3,4,5,6]
		combined_list = min_n, negative, epochs
		parameters=list(itertools.product(*combined_list))
		for i,j in enumerate(parameters):
			name='model'+str(j[0])+'-'+str(j[1])+'-'+str(j[2])
			model=FT_gensim.load(self.models+name+".model")
			print("Figure for model name %s" %name)
			self.FT_ec_tsne(model,name)
			self.FT_ec_umap(model,name)
			self.plot_umap(cluster_df=self.get_metacyc_df(),model_gensim=model,title_text=name)
			

if __name__=='__main__':
	Embeddings().plot_all_figures()


