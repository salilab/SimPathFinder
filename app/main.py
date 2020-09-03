from flask import Flask, flash, render_template, request, url_for, redirect
import jinja2
import os,sys
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import sqlite3 as sql
from dbconnect import create_users_table,create_login_table
import gensim
import pickle

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

@app.route("/")
def home():
    return render_template("layout.html")

@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/application/")
def application():
    if name and email:
        return render_template("application.html")
    else:
        return "<h1>Please fill your details to proceed.</h1>"

@app.route("/Similarity/")
def LDAResult():
    return render_template("Similarity.html")

@app.route("/Classification/")
def FTResult():
    return render_template("Classification.html")

if __name__ == "__main__":
    app.run(debug=True)
