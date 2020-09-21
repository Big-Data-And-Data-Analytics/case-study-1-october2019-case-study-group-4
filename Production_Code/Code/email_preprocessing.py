from .Preprocessing.read_csv import read_file
from .Preprocessing.split_emails import Preprocess
from .Preprocessing.create_textfile_content import create_textfile
from .Preprocessing.clean_content import CleanEmails
from .path_creation import OsPath

folder = OsPath()

class EmailPreprocessing:

#    def __init__(self):
#        self.emails = read_file(input("Enter the path for CSV file: "))

    def basic_analysis(self):
        try:
            folder.read_datafolder()
            data = folder.read_dir_files(None, 'Data')
            if data == None: data = input("Enter the path for CSV file: ")
            rows = int(input("Enter the nrows to be considered: "))
            emails = read_file(data,rows)
            print("Provided dataset has below columns")
            for key in emails.columns:
                print(key)
            inp_data = input("Please provide the data column to train the model:")
            print(f"Entered column name is {inp_data}")
            if inp_data not in emails.columns:
                print(f"Provided column name {inp_data} doesnot exist in the data")
                exit
            information = Preprocess()
            analysis = information.preprocessing_emails(emails, inp_data)
            return analysis, inp_data
        except ValueError:
            print("Wrong choice...")
            exit
        except FileNotFoundError:
            print("File not found in the directory")
            exit
    
    def create_text(self, input_dataframe):
        create_textfile(input_dataframe)

    def clean_text(self, input_dataframe, inp_data):
        clean_class = CleanEmails(input_dataframe, inp_data)
        clean_nlp_content = clean_class.clean_emails()
        return clean_nlp_content
