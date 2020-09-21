from .Machine_Learning.Training_ML import TrainML
from .path_creation import *
import pickle
import pandas as pd

folder = OsPath()

def train_ml_model():
    folder.clean_text_inputs()
    ml_input = folder.read_dir_files('.csv', 'Clean Text in Preprocessing')
    if ml_input==None: ml_input = input("Enter the .csv file with data to train email classification :\n")
    try:
        df = pd.read_csv(ml_input, engine='python')
    except FileNotFoundError:
        print("Given file doesnot exist in the location")
        exit
    print("Provided dataset has below columns")
    for key in df.columns:
        print(key)
    inp_data = input("Please provide the data column to train the model:")
    print(f"Entered column name is {inp_data}")
    if inp_data not in df.columns:
        print(f"Provided column name {inp_data} doesnot exist in the data")
        exit
    df.dropna(subset=[inp_data], inplace=True)
    # folder.reset_directory()
    folder.ml_model()
    train_ml_class = TrainML(df,inp_data)
    train_ml_class.train_ml()