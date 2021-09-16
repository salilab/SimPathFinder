# SimPathFinder

## Project objective 
- `1.` Predicting enzymatic pathway type/class from partially/fully annotated enzymes**.
- `2.` Identify similar, existing pathways from KEGG and MetaCyc, given a set of partially/fully annotated enzymes**. 

**from a prokaryotic/eukaryotic organism


## List of files and directories:

- `ecdata`     contains all relevant unlabeled data from BioCyc and MetaCyc (scrapped from webpages && downloaded from API) 

- `labeldata`   contains all relevant labeled data from KEGG and MetaCyc 

- `src`      contains all the modeling/scrapping scripts  

- `Flaskapp`  contains flask templates and scripts for webserver

- `tests`    contains unit tests

## List of classes:


### Classes for data extraction and preparation

- `ExtractUnlabeledData`  class to extract unlabeled data from BioCyc and MetaCyc
             
- `SampleUnlabeledData`  class to expand unlabeled data using resampling, child of `ExtractUnlabeledData` 

- `PathwayScrapper` class to extract labeled data from MetaCyc

- `ExtractLabeledData` class to process and clean labeled data from MetaCyc webpages
                  
- `BalanceLabelData` class to balance labeled data



### Classes to create embeddings 

- `CreateEmbedding` class to create fasttext embeddings 

- `ParameterizeEmbedding` class to explore hyperparameters for creating embeddigns, child of `CreateEmbedding`

- `ClusterEmbedding` class to cluster enzyme embedding vectors, child of `CreateEmbedding`

- `ClusterPWYEmbedding` class to cluster pathway embedding vectors, child of `CreateEmbedding`


### Classes to create ML models 

- `ModelVectors` class to get dependent and independent variables for building models

- `BuildClassicalModel` class to set up and run classical ML models using scikit-learn, child of `ModelVectors`

- `BuildControlModels` class to set up and run classical ML models on control datasets, child of `BuildClassicalModel`

- `BuildAnnot3Models` class to set up and run classical ML models on datasets build using 3 digit EC number annotations, child of `BuildClassicalModel` 

- `BuildAnnot2Models` class to set up and run classical ML models on datasets build using 2 digit EC number annotations, child of `BuildClassicalModel` 

- `BuildAnnot1Models` class to set up and run classical ML models on datasets build using 1 digit EC number annotations, child of `BuildClassicalModel`


### Classes to create DNN models 

- `DNNModel` class to create, parameterize and test LSTM based neural network model, child of `ModelVectors`


### Classes to analyze/measure results 

- `Metrics` class to define all required metrics 

- `EvaluateMetrics` class evaluate output from ML model, child of `Metrics` and `ModelVectors`

- `EvaluateControl` class evaluate output from ML model for control datasets, child of `EvaluateMetrics`

- `EvaluateAnnot3` class to evaluate output from ML model for datasets build using 3 digit EC number annotations, child of `EvaluateMetrics` 

- `EvaluateAnnot2` class to evaluate output from ML model for datasets build using 2 digit EC number annotations, child of `EvaluateMetrics` 

- `EvaluateAnnot1` class to evaluate output from ML model for datasets build using 1 digit EC number annotations, child of `EvaluateMetrics` 

- `combinedEvaluations` class to compare and contrast output from all datasets, child of `Metrics` and `ModelVectors`

- `MetricsDNN` class to define all required metrics for NN output

- `EvaluateMetricsDNN` class evaluate output from DNN model, child of `MetricsDNN` and `ModelVectors`

- `EvaluateMetricsAnnot3DNN` class evaluate output from DNN model on datasets build using 1 digit EC number annotations, child of `EvaluateMetricsDNN`


### Classes to explore model performance on KEGG dataset (validation)

- `ExtractKEGGData` class to extract and evaluate KEGG data

- `ExtractKEGGDataControl` class to evalate KEGG data on control models, child of `ExtractKEGGData`


## Test site 

- [`Test application`](https://modbase.compbio.ucsf.edu/SimPathFinder/) 

## Information

_`Paper`_: Predicting ontology of enzymatic pathways using transfer learning

_`Author(s)`_: Sai J. Ganesan, Andrej Sali

_`Citation`_: TBD


