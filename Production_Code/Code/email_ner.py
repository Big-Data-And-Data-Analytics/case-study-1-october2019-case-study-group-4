from .Feature_Engineering.training_ner import train_spacy
from .Feature_Engineering.evaluate_ner import model_evaluate
import pickle
from .path_creation import OsPath

folder = OsPath()
def train_ner_data():
    try:
        folder.read_datafolder()
        ner_input = folder.read_dir_files('.pickle', 'NER train data in Data')
        if ner_input == None: ner_input = input("Enter the pickle file to train NER :\n")
        ner_iter = int(input("Enter the number of iterations to train the NER model :"))
        TRAIN_DATA, nlp = train_spacy(ner_input,ner_iter,model=None)
        score = model_evaluate(nlp,TRAIN_DATA)
        print(f"Model evaluation scores are:\n{score}")
        return nlp
    except ValueError:
        print("Wrong choice...")
        exit
    except FileNotFoundError:
        print("File not found in the directory")
        exit

