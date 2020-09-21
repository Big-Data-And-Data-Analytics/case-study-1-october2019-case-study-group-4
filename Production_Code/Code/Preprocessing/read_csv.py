import pandas as pd
import os

def read_file(data,rows=None):
    extension = os.path.splitext(data)[-1]
    if extension == '.csv':
        if rows!=None: file = pd.read_csv(data, nrows=rows, engine='python')
        else:file = pd.read_csv(data, engine='python')
        print("CSV imported successfully \n")
        return file
    else:
        raise Exception("Enter a valid CSV file!")

    