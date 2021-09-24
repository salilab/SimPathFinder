from flask import Flask, flash, render_template, request, url_for, redirect
import jinja2
import os,sys
import matplotlib.pyplot as plt
#import sqlite3 as sql
#from dbconnect import create_users_table,create_login_table
import gensim
import pickle
sys.path.insert(0, "../src/pyext/")
sys.path.append('../..//models/')
from run_server import RunServerClassifier
import warnings


#############
#JINJA
#############
templateLoader = jinja2.FileSystemLoader(searchpath="./")
templateEnv = jinja2.Environment(loader=templateLoader)

def write_html(Template_Dict, template_file,output_file):
    template = templateEnv.get_template('templates/'+template_file)
    outputText=template.render(Template_Dict)
    with open(os.path.join('templates/',output_file),"w") as fh:
        fh.write(outputText)
#############

#############
#FLASK
#############
app = Flask(__name__)        
@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
@app.route("/home/")
def home():
    return render_template("application.html")

@app.route("/help/")
def help():
    return render_template("help.html")

@app.route("/application/")
def application():
    return render_template("application.html")

@app.route("/Similarity/",methods=['GET','POST'])
def Similarity():
    if request.method=='POST':
        enzyme_string=request.form['list']
        if enzyme_string:
            num,text1=RunServerClassifier().check_format(enzyme_string)
            print (num,text1)
            if num==1 :
                return "<h1>{}</h1>".format(text1)
            if num==0 : 
                try:
                    metacyc,kegg=RunServerClassifier().run_similarity(enzyme_string)
                    Template_Dict={}
                    Template_Dict['results_table_M']=metacyc
                    Template_Dict['results_table_K']=kegg
                    write_html(Template_Dict,"Similarity_temp.html","Similarity_Results.html")
                    return redirect(url_for('Similarity_Results'))
                except:
                    return "<h1>Unexpected error, please email ganesans@salilab.org for help.</h1>"

    return render_template("Similarity.html")

@app.route("/Classification/", methods=['GET','POST'])
def Classification():
    if request.method=='POST':
        enzyme_string=request.form['list']
        if enzyme_string:
            num,text1=RunServerClassifier().check_format(enzyme_string)
            print (num,text1)
            if num==1 :
                return "<h1>{}</h1>".format(text1)
            if num==0 : 
                try:
                    result_class,result_prob=RunServerClassifier().run_classifier(enzyme_string)
                    Template_Dict={}
                    Template_Dict['result_class']=result_class
                    Template_Dict['result_prob']=result_prob
                    write_html(Template_Dict,"Classification_temp.html","Classification_Results.html")
                    print ("written template")
                    return redirect(url_for('Classification_Results'))
                except:
                    return "<h1>Unexpected error, please email ganesans@salilab.org for help.</h1>"

    return render_template("Classification.html")

@app.route("/Classification_Results/")
def Classification_Results():
    return render_template("Classification_Results.html")

@app.route("/Similarity_Results/")
def Similarity_Results():
    return render_template("Similarity_Results.html")

if __name__ == "__main__":
    app.run(debug=True)
