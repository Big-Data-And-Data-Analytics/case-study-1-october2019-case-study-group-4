import pandas as pd    

def create_textfile(dataframe):
    file_name = str(input("Enter the File Name :"))
    dataframe.to_csv(file_name, header=True, index=None, mode='a')
