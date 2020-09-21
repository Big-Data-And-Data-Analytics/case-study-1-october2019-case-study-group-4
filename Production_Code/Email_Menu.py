from Code.Testing_Model import basic_analysis,begin_testing,view_email
from Code.email_preprocessing import *
from Code.email_nel import SelectionInputsForNel
from Code.path_creation import *
from Code.email_ner import train_ner_data
from Code.email_classif import train_ml_model

import os

folder_creation = OsPath()

def main_menu_options():
    print("\n")
    print("1. TEST")
    print("2. TRAIN")
    print("3. Exit")
    try:
        selection = int(input("Enter your choice: "))
        # folder_creation = OsPath()
        if selection == 1:
            print("\n")
            email_df = basic_analysis()
            emails = begin_testing(email_df)
            print(os.getcwd())
            print("\n")
            view_email(emails)
        elif selection == 2:
            print("\n")
            train_model()    
        elif selection == 3:
            print("\n")
            exit
        else:
            print("\n")
            print("Invalid Input")
            print("\n")
            exit
    #        main_menu_options()
    except ValueError:
        print("Wrong choice...")
        exit
        
def train_model():
    print("\n1. Preprocessing")
    print("2. Train NER")
    print("3. Train NEL")
    print("4. Train Classification")
    print("5. Exit")
    try:
        selection = int(input("Enter your choice: "))
        if selection == 1:
            pre_processing()
            print("\n")
        elif selection == 2:
            ner_menu_options()
            print("\n")
        elif selection == 3:
            nel_menu_options()
        elif selection == 4:
            print("ML")
            ml_menu_options()
            print("\n")
        elif selection == 5:
            print("\n")
            exit
            print("\n")
        else:
            print("/n")
            print("Enter Valid Options")
            exit
    except ValueError:
        print("Wrong choice...")
        exit 
    
def pre_processing():
    print("\n")
    print("1. Run PreProcessing and create inputs for NER and Classification")
    print("2. Create inputs for NER")
    print("3. Create inputs for Classification")
    print("4. Exit")
    try:
        selection = int(input("Enter your choice: "))
        print("\n")
        
        input_data = EmailPreprocessing()
        content_split, col = input_data.basic_analysis()
        
        if selection == 1:
            if 'email_body' in content_split.columns: data_column = content_split['email_body']
            else: data_column = content_split[col]
            folder_creation.ner_inputs_file()
            print("Generating file for the NER manual annotation in the below location...")
            print(os.getcwd())
            input_data.create_text(data_column)
            
            clean_body = input_data.clean_text(content_split, col)
            data_column = clean_body[[col,'clean_text']]
            folder_creation.clean_text_inputs()
            print("Generating file for the Similarity in the below location...")
            print(os.getcwd())
            input_data.create_text(data_column) 
            print("PreProcessed and created inputs for NER and Classification.")    
            print("\n")
            print("Exploratory data analysis Function is Pending")
            train_model()     
            # Database insertion function is pending
        elif selection == 2:
            if 'email_body' in content_split.columns: data_column = content_split['email_body']
            else: data_column = content_split[col]
            folder_creation.ner_inputs_file()
            input_data.create_text(data_column)
            print("Created inputs for NER.\n")
            train_model()
        elif selection == 3:
            clean_body = input_data.clean_text(content_split, col)
            data_column = clean_body[[col,'clean_text']]
            folder_creation.clean_text_inputs()
            input_data.create_text(data_column)
            print("Created inputs for Classification.\n")
            train_model()
        elif selection == 4:
            print("\n")
            exit
        else:
            print("\n")
            print("Enter Valid inputs of choice '1' or '2' or '3' or '4'")
            print("\n")
    except ValueError:
        print("Wrong choice...")
        exit

def nel_menu_options():
    print("\n1. Create the KnowledgeBase by using the customized NER model")
    print("2. Load the KnowledgeBase and Train the customized NER model")
    print("3. Exit")
    try:
        selection = int(input("Enter your choice: "))
        nelinputs = SelectionInputsForNel()
        if selection == 1:
            print("\n")
            nelinputs.create_kbase()
            print("Created KnowledgeBase by using the customized NER model")
            nel_menu_options()
        elif selection == 2:
            print("\n")
            nelinputs.set_kbase()
            print("Trained the customized NER model with KnowledgeBase")
            train_model()
        elif selection == 3:
            exit
        else:
            print("Invalid Input")
            print("\n")
            nel_menu_options()
    except ValueError:
        print("Wrong choice...")
        exit
        
def ner_menu_options():
    nlp = train_ner_data()
    folder_creation.ner_model()
    modelfile = input("Enter your Model Name: ")
    nlp.to_disk(modelfile)
    print("NER model successfully trained and saved to disk")
    train_model()

def ml_menu_options():
    folder_creation.ml_model()
    train_ml_model()
    main_menu_options()

main_menu_options()