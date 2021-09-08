from pathwayScrapper import PathwayScrapper

    link='https://biocyc.org/META/NEW-IMAGE?object='
    path='labeldata/'
    key='Biosynthesis'
    p_new={};e_new={};p_new[key]=[];e_new[key]=[]
    
    biosyn=PathwayScrapper().get_pflinks_biosynthesis(link,key,key,p_new,e_new)
    key='Energy-Metabolism'
    p_new[key]=[];e_new[key]=[]
    ener=PathwayScrapper().get_pflinks_energy(link,key,key,p_new,e_new)
    key='Activation-Inactivation-Interconversion'
    p_new[key]=[];e_new[key]=[]
    act=PathwayScrapper().get_pflinks_activation(link,key,key,p_new,e_new)
    key='Bioluminescence'
    p_new[key]=[];e_new[key]=[]
    bio=PathwayScrapper().get_pflinks_bio(link,key,key,p_new,e_new)
    key='Glycan-Pathways'
    p_new[key]=[];e_new[key]=[]
    gly=PathwayScrapper().get_pflinks_glycan(link,key,key,p_new,e_new)
    key='Macromolecule-Modification'
    p_new[key]=[];e_new[key]=[]
    mac=PathwayScrapper().get_pflinks_mac(link,key,key,p_new,e_new)
    key='Degradation'
    p_new[key]=[];e_new[key]=[]
    deg=PathwayScrapper().get_pflinks_deg(link,key,key,p_new,e_new)
    key='Detoxification'
    p_new[key]=[];e_new[key]=[]
    dox=PathwayScrapper().get_pflinks_dox(link,key,key,p_new,e_new)
    all_dicts=[biosyn,ener,act,bio,gly,mac,deg,dox]
    combined_dict=PathwayScrapper().combine_classes(all_dicts)
    pickle.dump(combined_dict, open(path+ "test.pkl", "wb" ) )
	data_df_multi=pickle.load(open(path+"test.pkl",'rb'))
