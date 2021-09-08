from bs4 import BeautifulSoup
import requests
import pickle
from re import search


class PathwayScrapper(object):
    def __init__(self):
        self.weblink_child = "https://biocyc.org"
        self.pathways = ['Activation-Inactivation-Interconversion', 'Bioluminescence', 'Biosynthesis',
                         'Degradation', 'Detoxification', 'Energy-Metabolism', 'Glycan-Pathways', 'Macromolecule-Modification',
                         'Metabolic-Clusters', 'Super-Pathways']
        self.pathways_dict = {i: j for i, j in enumerate(self.pathways)}

    def get_pflinks_biosynthesis(self, weblink, pathway_parent, pathway_child, p_dict, e_dict):
        page = requests.get(weblink+pathway_child)
        soup = BeautifulSoup(page.content, 'html.parser')
        page = soup.find_all('p')
        page_text = [p.getText() for p in page]
        parent_text = []
        for m, n in enumerate(page_text):
            if "Parent" in n:
                parent_text.append(n.replace('\n', '').replace(' ', ''))
        mydivs_p = []
        mydivs_e = []
        mydivs_p = [a['href']
                    for a in soup.findAll("a", {"class": "PATHWAY"}, href=True)]
        mydivs_e = [a['href'] for a in soup.findAll(
            "a", {"class": "ECOCYC-CLASS"}, href=True)]
        [p_dict[pathway_parent].append(
            i.split('/META/NEW-IMAGE?type=PATHWAY&object=')[1]) for i in mydivs_p]
        for i in mydivs_e:
            if (i not in e_dict[pathway_parent]) and ('NucleosideandNucleotideDegradation' not in parent_text[0]) and \
                ('Degradation/Utilization/Assimilation' not in parent_text[0]) and \
                ('Detoxification' not in parent_text[0]) and \
                    ('PolysaccharideDegradation' not in parent_text[0]):
                e_dict[pathway_parent].append(i)
                p_dict = self.get_pflinks_biosynthesis(
                    self.weblink_child, pathway_parent, i, p_dict, e_dict)
        return p_dict

    def get_pflinks_activation(self, weblink, pathway_parent, pathway_child, p_dict, e_dict):
        page = requests.get(weblink+pathway_child)
        soup = BeautifulSoup(page.content, 'html.parser')
        page = soup.find_all('p')
        page_text = [p.getText() for p in page]
        parent_text = []
        for m, n in enumerate(page_text):
            if "Parent" in n:
                parent_text.append(n)
        par = ''.join(parent_text)
        mydivs_p = []
        mydivs_e = []
        mydivs_p = [a['href']
                    for a in soup.findAll("a", {"class": "PATHWAY"}, href=True)]
        mydivs_e = [a['href'] for a in soup.findAll(
            "a", {"class": "ECOCYC-CLASS"}, href=True)]
        [p_dict[pathway_parent].append(
            i.split('/META/NEW-IMAGE?type=PATHWAY&object=')[1]) for i in mydivs_p]
        for i in mydivs_e:
            if i not in e_dict[pathway_parent] and 'Nitrogen Containing Glucoside Degradation' not in par:
                e_dict[pathway_parent].append(i)
                p_dict = self.get_pflinks_activation(
                    self.weblink_child, pathway_parent, i, p_dict, e_dict)
        return p_dict

    def get_pflinks_bio(self, weblink, pathway_parent, pathway_child, p_dict, e_dict):
        page = requests.get(weblink+pathway_child)
        soup = BeautifulSoup(page.content, 'html.parser')
        page = soup.find_all('p')
        page_text = [p.getText() for p in page]
        parent_text = []
        for m, n in enumerate(page_text):
            if "Parent" in n:
                parent_text.append(n)
        mydivs_p = []
        mydivs_e = []
        mydivs_p = [a['href']
                    for a in soup.findAll("a", {"class": "PATHWAY"}, href=True)]
        mydivs_e = [a['href'] for a in soup.findAll(
            "a", {"class": "ECOCYC-CLASS"}, href=True)]
        [p_dict[pathway_parent].append(
            i.split('/META/NEW-IMAGE?type=PATHWAY&object=')[1]) for i in mydivs_p]
        for i in mydivs_e:
            if i not in e_dict[pathway_parent]:
                e_dict[pathway_parent].append(i)
                p_dict = self.get_pflinks_bio(
                    self.weblink_child, pathway_parent, i, p_dict, e_dict)
        return p_dict

    def get_pflinks_deg(self, weblink, pathway_parent, pathway_child, p_dict, e_dict):
        page = requests.get(weblink+pathway_child)
        soup = BeautifulSoup(page.content, 'html.parser')
        page = soup.find_all('p')
        page_text = [p.getText() for p in page]
        parent_text = []
        for m, n in enumerate(page_text):
            if "Parent" in n:
                parent_text.append(n.replace('\n', '').replace(' ', ''))
        mydivs_p = []
        mydivs_e = []
        mydivs_p = [a['href']
                    for a in soup.findAll("a", {"class": "PATHWAY"}, href=True)]
        mydivs_e = [a['href'] for a in soup.findAll(
            "a", {"class": "ECOCYC-CLASS"}, href=True)]
        [p_dict[pathway_parent].append(
            i.split('/META/NEW-IMAGE?type=PATHWAY&object=')[1]) for i in mydivs_p]
        for i in mydivs_e:
            if (i not in e_dict[pathway_parent]) and ('Detoxification' not in parent_text[0]) and \
                ('Activation/Inactivation/Interconversion' not in parent_text[0]) and \
                ('GenerationofPrecursorMetabolitesandEnergy' not in parent_text[0]) and \
                    ('Biosynthesis' not in parent_text[0]):
                e_dict[pathway_parent].append(i)
                p_dict = self.get_pflinks_deg(
                    self.weblink_child, pathway_parent, i, p_dict, e_dict)
        return p_dict

    def get_pflinks_clus(self, weblink, pathway_parent, pathway_child, p_dict, e_dict):
        page = requests.get(weblink+pathway_child)
        print(weblink+pathway_child)
        soup = BeautifulSoup(page.content, 'html.parser')
        page = soup.find_all('p')
        page_text = [p.getText() for p in page]
        parent_text = []
        for m, n in enumerate(page_text):
            if "Parent" in n:
                parent_text.append(n.replace('\n', '').replace(' ', ''))
        print(parent_text)
        mydivs_p = []
        mydivs_e = []
        mydivs_p = [a['href']
                    for a in soup.findAll("a", {"class": "PATHWAY"}, href=True)]
        mydivs_e = [a['href'] for a in soup.findAll(
            "a", {"class": "ECOCYC-CLASS"}, href=True)]
        [p_dict[pathway_parent].append(
            i.split('/META/NEW-IMAGE?type=PATHWAY&object=')[1]) for i in mydivs_p]
        for i in mydivs_e:
            if (i not in e_dict[pathway_parent]) and ('Detoxification' not in parent_text[0]) and \
                ('Activation/Inactivation/Interconversion' not in parent_text[0]) and \
                ('GenerationofPrecursorMetabolitesandEnergy' not in parent_text[0]) and \
                ('Biosynthesis' not in parent_text[0]) and \
                ('PolysaccharideDegradation' not in parent_text[0]) and \
                    ('Degradation' not in parent_text[0]):
                e_dict[pathway_parent].append(i)
                p_dict = self.get_pflinks_clus(
                    self.weblink_child, pathway_parent, i, p_dict, e_dict)
        return p_dict

    def get_pflinks_super(self, weblink, pathway_parent, pathway_child, p_dict, e_dict):
        page = requests.get(weblink+pathway_child)
        print(weblink+pathway_child)
        soup = BeautifulSoup(page.content, 'html.parser')
        page = soup.find_all('p')
        page_text = [p.getText() for p in page]
        parent_text = []
        for m, n in enumerate(page_text):
            if "Parent" in n:
                parent_text.append(n.replace('\n', '').replace(' ', ''))
        print(parent_text)
        mydivs_p = []
        mydivs_e = []
        mydivs_p = [a['href']
                    for a in soup.findAll("a", {"class": "PATHWAY"}, href=True)]
        mydivs_e = [a['href'] for a in soup.findAll(
            "a", {"class": "ECOCYC-CLASS"}, href=True)]
        [p_dict[pathway_parent].append(
            i.split('/META/NEW-IMAGE?type=PATHWAY&object=')[1]) for i in mydivs_p]
        for i in mydivs_e:
            if (i not in e_dict[pathway_parent]) and ('Detoxification' not in parent_text[0]) and \
                ('Activation/Inactivation/Interconversion' not in parent_text[0]) and \
                ('GenerationofPrecursorMetabolitesandEnergy' not in parent_text[0]) and \
                ('Biosynthesis' not in parent_text[0]) and \
                ('PolysaccharideDegradation' not in parent_text[0]) and \
                    ('Degradation' not in parent_text[0]):
                e_dict[pathway_parent].append(i)
                p_dict = self.get_pflinks_clus(
                    self.weblink_child, pathway_parent, i, p_dict, e_dict)
        return p_dict

    def get_pflinks_dox(self, weblink, pathway_parent, pathway_child, p_dict, e_dict):
        page = requests.get(weblink+pathway_child)
        soup = BeautifulSoup(page.content, 'html.parser')
        page = soup.find_all('p')
        page_text = [p.getText() for p in page]
        parent_text = []
        for m, n in enumerate(page_text):
            if "Parent" in n:
                parent_text.append(n.replace('\n', '').replace(' ', ''))
        mydivs_p = []
        mydivs_e = []
        mydivs_p = [a['href']
                    for a in soup.findAll("a", {"class": "PATHWAY"}, href=True)]
        mydivs_e = [a['href'] for a in soup.findAll(
            "a", {"class": "ECOCYC-CLASS"}, href=True)]
        [p_dict[pathway_parent].append(
            i.split('/META/NEW-IMAGE?type=PATHWAY&object=')[1]) for i in mydivs_p]
        for i in mydivs_e:
            if (i not in e_dict[pathway_parent]) and ('Degradation/Utilization/Assimilation' not in parent_text[0]):
                e_dict[pathway_parent].append(i)
                p_dict = self.get_pflinks_dox(
                    self.weblink_child, pathway_parent, i, p_dict, e_dict)
        return p_dict

    def get_pflinks_mac(self, weblink, pathway_parent, pathway_child, p_dict, e_dict):
        page = requests.get(weblink+pathway_child)
        soup = BeautifulSoup(page.content, 'html.parser')
        page = soup.find_all('p')
        page_text = [p.getText() for p in page]
        parent_text = []
        for m, n in enumerate(page_text):
            if "Parent" in n:
                parent_text.append(n.replace('\n', '').replace(' ', ''))
        mydivs_p = []
        mydivs_e = []
        mydivs_p = [a['href']
                    for a in soup.findAll("a", {"class": "PATHWAY"}, href=True)]
        mydivs_e = [a['href'] for a in soup.findAll(
            "a", {"class": "ECOCYC-CLASS"}, href=True)]
        [p_dict[pathway_parent].append(
            i.split('/META/NEW-IMAGE?type=PATHWAY&object=')[1]) for i in mydivs_p]
        for i in mydivs_e:
            if (i not in e_dict[pathway_parent]):
                e_dict[pathway_parent].append(i)
                p_dict = self.get_pflinks_mac(
                    self.weblink_child, pathway_parent, i, p_dict, e_dict)
        return p_dict

    def get_pflinks_glycan(self, weblink, pathway_parent, pathway_child, p_dict, e_dict):
        page = requests.get(weblink+pathway_child)
        soup = BeautifulSoup(page.content, 'html.parser')
        page = soup.find_all('p')
        page_text = [p.getText() for p in page]
        parent_text = []
        for m, n in enumerate(page_text):
            if "Parent" in n:
                parent_text.append(n.replace('\n', '').replace(' ', ''))
        mydivs_p = []
        mydivs_e = []
        mydivs_p = [a['href']
                    for a in soup.findAll("a", {"class": "PATHWAY"}, href=True)]
        mydivs_e = [a['href'] for a in soup.findAll(
            "a", {"class": "ECOCYC-CLASS"}, href=True)]
        [p_dict[pathway_parent].append(
            i.split('/META/NEW-IMAGE?type=PATHWAY&object=')[1]) for i in mydivs_p]
        for i in mydivs_e:
            if (i not in e_dict[pathway_parent]) and ('Detoxification' not in parent_text[0]) and \
                    ('Degradation/Utilization/Assimilation' not in parent_text[0]) and \
                    ('Biosynthesis' not in parent_text[0]):
                e_dict[pathway_parent].append(i)
                p_dict = self.get_pflinks_glycan(
                    self.weblink_child, pathway_parent, i, p_dict, e_dict)
        return p_dict

    def get_pflinks_energy(self, weblink, pathway_parent, pathway_child, p_dict, e_dict):
        page = requests.get(weblink+pathway_child)
        soup = BeautifulSoup(page.content, 'html.parser')
        page = soup.find_all('p')
        page_text = [p.getText() for p in page]
        parent_text = []
        for m, n in enumerate(page_text):
            if "Parent" in n:
                parent_text.append(n.replace('\n', '').replace(' ', ''))
        mydivs_p = []
        mydivs_e = []
        mydivs_p = [a['href']
                    for a in soup.findAll("a", {"class": "PATHWAY"}, href=True)]
        mydivs_e = [a['href'] for a in soup.findAll(
            "a", {"class": "ECOCYC-CLASS"}, href=True)]
        [p_dict[pathway_parent].append(
            i.split('/META/NEW-IMAGE?type=PATHWAY&object=')[1]) for i in mydivs_p]
        for i in mydivs_e:
            if (i not in e_dict[pathway_parent]) and ('InorganicNutrientMetabolism' not in parent_text[0]) and \
                    ('Degradation' not in parent_text[0]) and \
                    ('Biosynthesis' not in parent_text[0]) and \
                    ('SulfurCompoundMetabolism' not in parent_text[0]):
                e_dict[pathway_parent].append(i)
                p_dict = self.get_pflinks_energy(
                    self.weblink_child, pathway_parent, i, p_dict, e_dict)
        return p_dict

    def combine_classes(self, all_dicts):
        relevant_pwy = {}
        for i in all_dicts:
            relevant_pwy.update(i)
        return relevant_pwy

    def clean_classes(self, class_dict):
        comb_list = []
        new_class_dict = {}
        for i, j in class_dict.items():
            if i == 'Bioluminescence':
                comb_list = comb_list+j
                new_class_dict[i] = j
            if i == 'Activation-Inactivation-Interconversion':
                comb_list = comb_list+j
                new_class_dict[i] = j
            if i == 'Glycan-Pathways':
                comb_list = comb_list+j
                new_class_dict[i] = j
            if i == 'Macromolecule-Modification':
                j_new = list(set(j)-set(comb_list))
                comb_list = comb_list+j_new
                new_class_dict[i] = j_new
            if i == 'Detoxification':
                j_new = list(set(j)-set(comb_list))
                comb_list = comb_list+j_new
                new_class_dict[i] = j_new
            if i == 'Degradation':
                j_new = list(set(j)-set(comb_list))
                comb_list = comb_list+j_new
                new_class_dict[i] = j_new
            if i == 'Biosynthesis':
                j_new = list(set(j)-set(comb_list))
                comb_list = comb_list+j_new
                new_class_dict[i] = j_new
                break
        return new_class_dict

    def get_intersection(self, a, b):
        return list(set(a).intersection(set(b)))

    def check_pwy_intersection(self, dictionary_of_interest):
        for i, j in dictionary_of_interest.items():
            for m, n in dictionary_of_interest.items():
                if i != m:
                    print(i, m, len(list(set(j).intersection(set(n)))))
