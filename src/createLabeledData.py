from __init__ import ExtractLabeledData

E = ExtractLabeledData(data_dir='../labeldata/')
E.get_pathways()
E.get_pathway_names()
E.get_classes_dict()
E.create_df_all_labels()
