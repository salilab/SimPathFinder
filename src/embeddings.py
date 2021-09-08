import pickle
import numpy as np
import fasttext
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from gensim import corpora, models
from gensim.models.fasttext import FastText as FT_gensim
import itertools
import umap
sns.set(style='white', rc={'figure.figsize': (10, 8)})
sns.set_style("whitegrid")


class CreateEmbedding(object):
    def __init__(self, data_dir='../ecdata/', EC=[], model_dir='models/', *args, **kwargs):
        self.data_dir = data_dir
        self.EC = EC
        self.model_dir = model_dir

    def FT_sg(self, epoch=5, negative=20, min_n=1, size=300, window=5):
        model_gensim = FT_gensim(size=size)
        model_gensim.build_vocab(sentences=self.EC, min_count=1)
        model_gensim.train(self.EC,
                           total_examples=len(self.EC),
                           epochs=epoch,
                           min_n=min_n,
                           window=window,
                           sg=1,
                           negative=negative)
        model_gensim.save("tierT12_10_"+str(size)+".model")
        return model_gensim

    def preLoadEmbedding(self, modelName='tierT12_10_300'):
        self.model = FT_gensim.load(self.model_dir + modelName+'.model')
        return self.model


class ParameterizeEmbedding(CreateEmbedding):
    def __init__(self):
        super().__init__(data_dir='../ecdata/', EC=[])

    def measure_results_all(self, model, keys=['ec:1', 'ec:2', 'ec:3', 'ec:4', 'ec:5', 'ec:6']):
        count = []
        mm = []

        def measure_results(model, key):
            vals = model.wv.most_similar(key, topn=200)
            count = 0
            min_sim = 1
            max_sim = 0
            for enzyme, sim in vals:
                if key in enzyme:
                    count += 1
                min_sim = min(round(sim, 2), min_sim)
                max_sim = max(round(sim, 2), max_sim)
            return count, (min_sim, max_sim)

        for key in keys:
            c, m = measure_results(model, key)
            count.append(c)
            mm.append(m)
        return count, mm

    def parameterize_FT(self, epochs=range(5, 60, 10), size=[100, 300]):
        combined_list = epochs, size
        parameters = list(itertools.product(*combined_list))
        lst = []
        for i, j in enumerate(parameters):
            self.model = self.FT_sg(epoch=j[0], size=j[1])
            count, mm = self.measure_results_all(model=self.model)
            lst.append([j[0], j[1], count[0], mm[0], count[1], mm[1], count[2], mm[2], count[3],
                        mm[3], count[4], mm[4], count[5], mm[5]])
            parameters = pd.DataFrame(lst, columns=['epoch', 'vec size', 'ec:1 sim', 'ec:1 mm',
                                                    'ec:2 sim', 'ec:2 mm', 'ec:3 sim', 'ec:3 mm', 'ec:4 sim', 'ec:4 mm', 'ec:5 sim', 'ec:5 mm',
                                                    'ec:6 sim', 'ec:6 mm'])
        return parameters


class ClusterEmbedding(CreateEmbedding):
    def __init__(self):
        super().__init__(data_dir='../ecdata/', EC=[])

    def get_clusters(self, model, keys=['ec:1', 'ec:2', 'ec:3', 'ec:4', 'ec:5', 'ec:6']):
        embedding_clusters = []
        word_clusters = []
        for word in keys:
            embeddings = []
            words = []
            for similar_word, _ in model.wv.most_similar(word, topn=100):
                words.append(similar_word)
                embeddings.append(model.wv[similar_word])
            embedding_clusters.append(embeddings)
            word_clusters.append(words)
        return word_clusters, embedding_clusters

    def plot_similar_words(self, labels, embedding_clusters, word_clusters):
        # fig
        plt.figure(figsize=(7, 5))
        ax = plt.subplot(111)
        colors = sns.color_palette("Set2")
        for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
            x = embeddings[:, 0]
            y = embeddings[:, 1]
            plt.scatter(x, y, c=[color], label=label, s=600, edgecolor='gray',)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Dimension #1', size=16)
        plt.ylabel('Dimension #2', size=16)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0., fontsize=16)
        # plt.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig("tierT12_10_.png")
        plt.show()

    def FT_tsne(self, model, keys=['ec:1', 'ec:2', 'ec:3', 'ec:4', 'ec:5', 'ec:6']):
        word, embeddings = self.get_clusters(model)
        embedding_clusters = np.array(embeddings)
        n, m, k = embedding_clusters.shape
        tsne_model_en_2d = TSNE(
            perplexity=35, n_components=2, init='pca', n_iter=5000, random_state=32)
        X_2d = np.array(tsne_model_en_2d.fit_transform(
            embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
        self.plot_similar_words(keys, X_2d, word)


class ClusterPWYEmbedding(ClusterEmbedding):
    def __init__(self, data_dir='../labeldata/',
                 model_dir='../models/',
                 metacyc='LabeledData.pkl',
                 kegg='df_kegg.pkl'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.metacyc = metacyc
        self.kegg = kegg
        self.categories = ['Degradation', 'Biosynthesis', 'Energy',
                           'Activation', 'Glycan', 'Macromolecule', 'Detoxification']
        self.kegg_cats = ['Carbohydrate-metabolism',
                          'Glycan-biosynthesis-and-metabolism']
        self.df = pickle.load(open(self.data_dir + self.metacyc, "rb"))
        self.df_kegg = pickle.load(open(self.data_dir+self.kegg, "rb"))

        self.model = self.preLoadEmbedding()
        super().__init__()

    def plot_umap(self, n_neighbors=100, min_dist=0.1):
        fig = plt.figure(figsize=(6, 5))
        ax1 = fig.add_subplot(111)
        colors = sns.color_palette("Set2")
        for ind, label in enumerate(self.categories):
            labels = self.df[self.df[label] == 1]  # .value_counts()
            embeddings = []
            PWYs = labels.EC.to_list()
            for PWY in PWYs:
                embeddings.append(np.mean(self.model.wv[PWY], axis=0))
            embedding_clusters = np.array(embeddings)
            x, y = embedding_clusters.shape
            X_2d = umap.UMAP(random_state=2, n_neighbors=n_neighbors,
                             min_dist=min_dist).fit_transform(embedding_clusters)
            # colors = 'r', 'g', 'b', 'orange', 'pink', 'm','purple','brown','y'
            ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=colors[ind],
                        s=200, edgecolor='gray', label=label)
        plt.xlabel('Dimension 1', size=18, color='black')
        plt.ylabel('Dimension 2', size=18, color='black')
        plt.ylim([-7.5, 12.5])
        plt.grid(None)
        plt.rcParams['axes.grid'] = False
        ax1.tick_params(axis='both', which='major', labelsize=0)
        plt.grid(None)
        plt.show()
        plt.tight_layout()

    def clean_df(self,):
        self.df_kegg['Label Name'] = self.df_kegg['Label Name'].str.lower()
        self.df_kegg['Name'] = self.df_kegg['Name'].str.lower()

        def f(word, tag='biosyn'):
            if tag in word:
                return tag
            else:
                return 'None'

        self.df_kegg['biosyn'] = self.df_kegg['Name'].apply(f)
        self.df_kegg['deg'] = self.df_kegg['Name'].apply(f, tag='deg')
        self.df_kegg['glycan'] = self.df_kegg['Label Name'].apply(
            f, tag='glycan')
        self.df_kegg['carb'] = self.df_kegg['Label Name'].apply(f, tag='carb')

    def plot_umap_kegg(self, title_text='blah', n_neighbors=20, min_dist=0.1):
        fig = plt.figure(figsize=(6, 5))
        ax1 = fig.add_subplot(111)
        colors = sns.color_palette("pastel")

        for ind, label in enumerate(self.kegg_cats):
            # .value_counts()
            labels = self.df_kegg[self.df_kegg['Label Name'] == label]
            embeddings = []
            PWYs = labels.EC.to_list()

            for PWY in PWYs:
                embeddings.append(np.mean(self.model.wv[PWY], axis=0))
            embedding_clusters = np.array(embeddings)
            x, y = embedding_clusters.shape
            X_2d = umap.UMAP(random_state=2, n_neighbors=n_neighbors,
                             min_dist=min_dist).fit_transform(embedding_clusters)
            # colors = 'r', 'g', 'b', 'orange', 'pink', 'm','purple','y','red','blue','green','black'
            ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=colors[ind],
                        s=200, edgecolor='gray', label=label)
        plt.xlabel('Dimension 1', size=18, color='black')
        plt.ylabel('Dimension 2', size=18, color='black')
        plt.legend(fontsize=12)
        plt.rcParams['axes.grid'] = False
        plt.grid(None)
        ax1.tick_params(axis='both', which='major', labelsize=0)
        plt.grid(None)
        plt.show()
        plt.tight_layout()

    def plot_umap_kegg_val(self, title_text='blah', n_neighbors=20, min_dist=0.1):
        fig = plt.figure(figsize=(6, 5))
        ax1 = fig.add_subplot(111)
        colors = sns.color_palette("pastel")
        self.clean_df()
        cats = ['biosyn', 'deg', 'glycan', 'carb']
        for ind, label in enumerate(cats):
            labels = self.df_kegg[self.df_kegg[label]
                                  == label]  # .value_counts()
            embeddings = []
            PWYs = labels.EC.to_list()

            for PWY in PWYs:
                embeddings.append(np.mean(self.model.wv[PWY], axis=0))
            embedding_clusters = np.array(embeddings)
            x, y = embedding_clusters.shape
            X_2d = umap.UMAP(random_state=2, n_neighbors=n_neighbors,
                             min_dist=min_dist).fit_transform(embedding_clusters)
            # colors = 'r', 'g', 'b', 'orange', 'pink', 'm','purple','y','red','blue','green','black'
            ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=colors[ind],
                        s=200, edgecolor='gray', label=label)
        plt.xlabel('Dimension 1', size=18, color='black')
        plt.ylabel('Dimension 2', size=18, color='black')
        plt.rcParams['axes.grid'] = False
        plt.grid(None)
        ax1.tick_params(axis='both', which='major', labelsize=0)
        plt.grid(None)
        plt.show()
        plt.tight_layout()


class ClusterEmbeddingDN(ClusterEmbedding):
    def __init__(self):
        super().__init__(data_dir='../ecdata/', EC=[])
