from .Feature_Engineering.training_nel import NelTraining
from .path_creation import OsPath
import pandas as pd

folder = OsPath()

class SelectionInputsForNel():

    def __init__(self):
        folder.ner_model()
        inp = folder.ner_folder()
        if inp == None: self.custom_ner_model = input("Enter the SpaCy Cutomized NER Model Path:\n")
        else: self.custom_ner_model = inp
        self.train_nel = NelTraining(self.custom_ner_model)
    
    def create_kbase(self):
        folder.read_datafolder()
        pickle_file = folder.read_dir_files('.pickle', 'NER train data in Data')
        if pickle_file == None: pickle_file = input("Enter the NER train data Pickle File:\n")
        self.train_nel.creating_knowledge(pickle_file)

    def set_kbase(self):
        folder.read_preinput()
        ner_kb_inputdata = folder.read_dir_files('.csv', 'Knowledge Base in Pre_input')
        if ner_kb_inputdata == None: ner_kb_inputdata = input("Enter the Knowledge Base file:\n")
        names =  pd.read_csv(ner_kb_inputdata, sep=';')
        folder.read_preinput()
        train_input_file = folder.read_dir_files('.csv', 'training knowledge base along with NER in Pre_input')
        if train_input_file == None: train_input_file = input("Enter the Training file:\n")
        train_data_2 =  pd.read_csv(train_input_file, sep=';')
        self.train_nel.settingup_knowledgebase(names,train_data_2)