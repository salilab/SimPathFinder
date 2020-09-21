# SimPathFinder

## Project objective 
- `1.` Predicting enzymatic pathway type/class from partially/fully annotated enzymes**.
- `2.` Identify similar, existing pathways from KEGG and MetaCyc, given a set of partially/fully annotated enzymes**. 

**from a prokaryotic/eukaryotic organism


## List of files and directories:

- `data`     contains all relevant data from KEGG and MetaCyc (scrapped from webpages && downloaded from API) 

- `src`      contains all the modeling/scrapping scripts  

- `EDA`      contains an explaratory data analysis script

- `app`      contains flask templates and scripts for webserver

- `tests`    contains unit tests

## List of classes:

- `PathwayClassifier`  class to predict pathway class
             
- `PathwaySimilarity`  class to predict similar pathways from database 

- `EnsembleClassifier` class to combine multiple classifiers
                  
- `Embeddings` class to create FastText embeddings

- `PathwayDF` class to create combined dataframes from scrapped data

- `PathwayScrapper` class to scrap pathway ontology and other information from MetaCyc 

## Test site 

- [`Test application`](https://modbase.compbio.ucsf.edu/SimPathFinder/) 

## Information

_Author(s)_: Sai J. Ganesan


