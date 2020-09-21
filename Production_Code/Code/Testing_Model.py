from .Preprocessing.read_csv import read_file
from .Preprocessing.split_emails import Preprocess
from .Preprocessing.clean_content import CleanEmails
from .Testing.ner_frame import ner_df
from .Testing.similarity import Activity
from .Testing.classification_test import test_cluster_model
from .path_creation import OsPath

folder = OsPath()
def basic_analysis():
    try:
        folder.read_datafolder()
        data = folder.read_dir_files(None, 'Data')
        if data == None: data = input("Enter the path for CSV file: ")    
        rows = int(input("Enter the nrows to be considered: "))
        # print(rows)
        emails = read_file(data,rows)
        print("Provided dataset has below columns")
        for key in emails.columns:
            print(key)
        inp_data = input("Please provide the data column to test the model:")
        print(f"Entered column name is {inp_data}")
        if inp_data not in emails.columns:
            print(f"Provided column name {inp_data} doesnot exist in the data")
            exit
        information = Preprocess()
        analysis = information.preprocessing_emails(emails, inp_data)
        clean_class = CleanEmails(emails, inp_data)
        clean_nlp_content = clean_class.clean_emails()
        return analysis
    except ValueError:
        print("Wrong choice...")
        exit
    except FileNotFoundError:
        print("File not found in the directory")
        exit

def begin_testing(email_df):
    folder.read_datafolder()
    inp_file = folder.read_dir_files(None, 'Generic Phrases in Data')
    extraction_2 = Activity(inp_file)
    extraction_2.activity_entity(email_df)
    folder.ml_model()
    ml_model = folder.read_dir_files('.sav', 'email classification in Machine learning')
    test_cluster_model(ml_model, email_df)
    folder.nel_kb_vocab()
    ner_model = folder.ner_folder()
    if ner_model == None:
        folder.ner_model()
        ner_model = folder.ner_folder()
    if ner_model == None:  ner_model = input("Enter your trained model:\n")
    email = ner_df(ner_model, email_df)
    # email.drop(['file','message','tokens','clean_text', 'Conent_NER'], axis=1, inplace=True)
    filename = str(input("Enter the File name for the final output with extension .csv: \n"))
    folder.final_output()
    email.to_csv(filename, index = False, header = True)
    return email

def view_email(email_df):
    while True:
        try:
            ind = input("\nEnter the index of the mail to view entire details of email classification. If you want to exit enter 'y':\n")
            if ind.lower() == 'y': exit()
            else: ind = int(ind)
            print("\nDate:\n",email_df['Date'][ind])
            print("\nSubject:\n",email_df['Subject'][ind])
            print("\nSender:\n",email_df['Sender'][ind])
            print("\nReceiver:\n",email_df['Receiver'][ind])
            print("\nBody of the Mail:\n",email_df['email_body'][ind])
            print("\nActivity:\n",email_df['activity'][ind])
            print("\nCluster:\n",email_df['cluster'][ind])
            print("\nNamed Entity Recognition:\n",email_df['NER_Info'][ind])
        except KeyError:
            print("Given index doesnot exist in the dataset")
            continue
        except ValueError:
            print("Please enter the correct index")
            continue

