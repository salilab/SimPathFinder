import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import os
import pickle
from collections import Counter
import matplotlib.cm as cm
sns.set(style='white', rc={'figure.figsize':(10,8)})
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings('ignore')

###Variables
global EC_dict,Brenda_numbers
EC_dict={'ec:1':'Oxidoreductases','ec:2':'Transferases',
         'ec:3':'Hydrolases','ec:4':'Lyases',
         'ec:5': 'Isomerases','ec:6': 'Ligases','ec:7':'Translocases'}

Brenda_numbers={'ec:1':9655,'ec:2':6622,'ec:3':10604,'ec:4':5111,'ec:5': 2083, 'ec:6': 1547, 'ec:7':966}

### useful functions

def merge_two_dict(dictionary1,dictionary2):
    dictionary3={get_key_from_value(dictionary2,k):v for k,v in list(dictionary1.items())}
    return dictionary3

def get_sorted_dict(dictionary):
    sorted_dict= sorted(dictionary.items(), key=lambda x: x[1],reverse=True)
    return sorted_dict

def get_key_from_value(dictionary,val):
    keys=list(dictionary.keys())
    vals=list(dictionary.values())
    return keys[vals.index(val)]

def merge_two_dict_key(dictionary1,dictionary2):
    dictionary3={dictionary2[k]:v for k,v in list(dictionary1.items())}
    return dictionary3

def convert_text(Name):
    return Name.replace('<i>','').replace('</i>','')

def get_enzyme_no_type_df(df,EC_dict):
    enzymes=df['EC'].to_list()
    all_enzymes=[j[:4] for i in enzymes for j in i]
    plot_enz=pd.Series(all_enzymes).value_counts()
    plot_enz_df=pd.DataFrame([plot_enz.values,list(EC_dict.values())])
    plot_enz_all=plot_enz_df.T
    plot_enz_all.rename(columns={0: 'Numbers',1:'EC_type'},inplace=True)
    plot_enz_all
    return plot_enz_all

def get_wordcloud(enz_list):
    word_cloud_dict=Counter(enz_list)
    wordcloud = WordCloud(width = 2000, height = 1000,background_color='white').generate_from_frequencies(word_cloud_dict)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis("off")
    #plt.savefig(Images+'EC_wc.png',type='png')
    
def get_wordcloud_big(enz_list):
    word_cloud_dict=Counter(enz_list)
    wordcloud = WordCloud(width = 2000, height = 1000).generate_from_frequencies(word_cloud_dict)
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis("off")

    
def get_all_enzymes(df):
    enzymes=df['EC'].to_list()
    all_enzymes=[j for i in enzymes for j in i]
    return all_enzymes

def get_hist(x,y,df):
    plt.figure(figsize=(5,5))
    g=sns.barplot(x,y,data=df,)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.5)
    g.set_xticklabels(g.get_xticklabels(),rotation=90,color='k');
    g.set_xlabel('');
    g.set_ylabel('Frequency',fontsize=20,color='k');
    sns.set_style("whitegrid")

def get_hist_dist(x,df):
    plt.figure(figsize=(5,5))
    g=sns.barplot(x,y,data=df,)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.5)
    g.set_xticklabels(g.get_xticklabels(),rotation=90,color='k',fontsize=20);
    g.set_xlabel('');
    g.set_ylabel('Frequency',fontsize=20,color='k');
    sns.set_style("whitegrid")
