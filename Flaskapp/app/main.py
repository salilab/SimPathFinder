from flask import Flask, flash, render_template, request, url_for, redirect, session, render_template_string
from flask_session import Session

import jinja2
import os,sys
import matplotlib.pyplot as plt
#import sqlite3 as sql
#from dbconnect import create_users_table,create_login_table
import gensim
import pickle
import random
sys.path.insert(0, "../src/pyext/")
sys.path.append('../..//models/')
from run_server import RunServerClassifier
import warnings
from datetime import timedelta


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
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=5)


Session(app)

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
@app.route("/application/")

def home():
    return render_template("application.html")

@app.route("/help/")
def help():
    return render_template("help.html")

@app.route("/Similarity/",methods=['GET','POST'])
def Similarity():
    if request.method=='POST':
        enzyme_string=request.form['list']
        session['enzyme']=request.form['list']
        session['id']=random.getrandbits(32)
        if session.get("enzyme"):
            num,text1=RunServerClassifier().check_format(enzyme_string)
            if num==1 :
                return "<h1>{}</h1>".format(text1)
            if num==0 : 
                try:
                    session['metacyc'],session['kegg']=RunServerClassifier().run_similarity(enzyme_string)
                    Template_Dict={}
                    Template_Dict['enzyme']=session['enzyme']
                    Template_Dict['results_table_M']=session['metacyc']
                    Template_Dict['results_table_K']=session['kegg']
                    print (Template_Dict)
                    write_html(Template_Dict,"Similarity_temp.html","Similarity_Results.html")

                    return redirect(url_for('Similarity_Results',id=session['id']))
                except:
                    return "<h1>Unexpected error, please email ganesans@salilab.org for help.</h1>"
        else:
            return render_template_string("""
<!DOCTYPE html>
<html lang="en" dir="ltr">

<head>
    <html xmlns="http://www.w3.org/1999/xhtml" lang="en-US" xml:lang="en-US">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta charset="utf-8">
    <title>Flask Parent Template</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/bootstrap.min.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/salilab.css') }}">
</head>

<body>
    <header>
        <div class="container">
            <div class="card border" style="border-color: solid black; margin-top:20px">
                <div class="card-body" style="background-color: white; height: 10rem;">
                    <div id="header1">
                        <h1 class="logo" align='center' class='text-center' style="border: none; margin-top: 2px; margin-right: 2px; margin-left: 2px;"><a href={{ url_for('home') }}> SimPathFinder </a></h1>
                    </div>
                    <div id="header1">
                        <h3 class="logo" align='center' class='text-center' style="border: none; margin-top: 2px; margin-right: 2px; margin-left: 2px;"> Find pathway ontology and other similar pathways in KEGG and MetaCyc</h3>
                    </div>
    </header>
    <div class="container">
        <div id="navigation_lab">
            <a href="//salilab.org/">Sali Lab Home</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/modweb/">ModWeb</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/modbase/">ModBase</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/evaluation/">ModEval</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/peptide/">PCSS</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/foxs/">FoXS</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//integrativemodeling.org/">IMP</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/multifit/">MultiFit</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/modpipe/">ModPipe</a>
            &nbsp;&nbsp;&nbsp;
        </div>
    </div>
    <div class="container">
        <div id="navigation_second">
            <a href="https://modbase.compbio.ucsf.edu/account/">ModBase Login</a>
            &nbsp;&nbsp;&nbsp;
            <a href="/home/">Web Server</a>
            &nbsp;&nbsp;&nbsp;
        </div>
    </div>
    </div>
    </div>

    <body>
        <div class="container">
            <div class="card border" style="border-color: solid black; margin-top:20px">
                <div class="card-body" style="background-color: white; height: 35rem;">
                    <h3>Error</h3>
                    <p align='center'>Enzyme list was not entered, please try again.</p>
                </div>
            </div>
        </div>
    </body>

   </body>
<footer>
   <hr size="2" width="80%">
    <div class="address">
        <p class="logo1" align='center' class='text-center' style="border: none; margin-bottom: 5px; margin-top: 5px; margin-right: 0px; margin-left: 0px;">Contact: <a href="ganesans@salilab.org">ganesans@salilab.org</a></p>
    </div>
</footer>
</html>
        """)


    return render_template("Similarity.html")

@app.route("/Classification/", methods=['GET','POST'])
def Classification():
    if request.method=='POST':
        enzyme_string=request.form['list']
        session['enzyme']=request.form['list']
        session['id']=random.getrandbits(32)
        if enzyme_string:
            num,text1=RunServerClassifier().check_format(enzyme_string)
            print (num,text1)
            if num==1 :
                return "<h1>{}</h1>".format(text1)
            if num==0 : 
                try:
                    session['result_class'],session['result_prob']=RunServerClassifier().run_classifier(enzyme_string)
                    Template_Dict={}
                    Template_Dict['result_class']=session['result_class']
                    Template_Dict['result_prob']=session['result_prob']
                    Template_Dict['enzyme']=session['enzyme']
                    print (Template_Dict)
                    write_html(Template_Dict,"Classification_temp.html","Classification_Results.html")
                    return redirect(url_for('Classification_Results',id=session['id']))
                except:
                    return "<h1>Unexpected error, please email ganesans@salilab.org for help.</h1>"
        else:
            return render_template_string("""
<!DOCTYPE html>
<html lang="en" dir="ltr">

<head>
    <html xmlns="http://www.w3.org/1999/xhtml" lang="en-US" xml:lang="en-US">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta charset="utf-8">
    <title>Flask Parent Template</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/bootstrap.min.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/salilab.css') }}">
</head>

<body>
    <header>
        <div class="container">
            <div class="card border" style="border-color: solid black; margin-top:20px">
                <div class="card-body" style="background-color: white; height: 10rem;">
                    <div id="header1">
                        <h1 class="logo" align='center' class='text-center' style="border: none; margin-top: 2px; margin-right: 2px; margin-left: 2px;"><a href={{ url_for('home') }}> SimPathFinder </a></h1>
                    </div>
                    <div id="header1">
                        <h3 class="logo" align='center' class='text-center' style="border: none; margin-top: 2px; margin-right: 2px; margin-left: 2px;"> Find pathway ontology and other similar pathways in KEGG and MetaCyc</h3>
                    </div>
    </header>
    <div class="container">
        <div id="navigation_lab">
            <a href="//salilab.org/">Sali Lab Home</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/modweb/">ModWeb</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/modbase/">ModBase</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/evaluation/">ModEval</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/peptide/">PCSS</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/foxs/">FoXS</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//integrativemodeling.org/">IMP</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/multifit/">MultiFit</a>
            &nbsp;&nbsp;&nbsp;
            <a href="//salilab.org/modpipe/">ModPipe</a>
            &nbsp;&nbsp;&nbsp;
        </div>
    </div>
    <div class="container">
        <div id="navigation_second">
            <a href="https://modbase.compbio.ucsf.edu/account/">ModBase Login</a>
            &nbsp;&nbsp;&nbsp;
            <a href="/home/">Web Server</a>
            &nbsp;&nbsp;&nbsp;
        </div>
    </div>
    </div>
    </div>

    <body>
        <div class="container">
            <div class="card border" style="border-color: solid black; margin-top:20px">
                <div class="card-body" style="background-color: white; height: 35rem;">
                    <h3>Error</h3>
                    <p align='center'>Enzyme list was not entered, please try again.</p>
                </div>
            </div>
        </div>
    </body>

   </body>
<footer>
   <hr size="2" width="80%">
    <div class="address">
        <p class="logo1" align='center' class='text-center' style="border: none; margin-bottom: 5px; margin-top: 5px; margin-right: 0px; margin-left: 0px;">Contact: <a href="ganesans@salilab.org">ganesans@salilab.org</a></p>
    </div>
</footer>
</html>
        """)
    return render_template("Classification.html")

@app.route("/<id>/Classification_Results.html")
def Classification_Results(id):
    return render_template("Classification_Results.html")


@app.route("/<id>/Similarity_Results.html")
def Similarity_Results(id):
    return render_template("Similarity_Results.html")


if __name__ == "__main__":
    app.run(debug=True)
