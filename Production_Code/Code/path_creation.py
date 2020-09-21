import os

class OsPath():

    def __init__(self):
        self.curdir = os.getcwd()
        self.reset = os.getcwd()

    def training_input(self):
        os.chdir(self.curdir)
        inp_path = "Train/Training_Input"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir

    def db_eda(self):
        os.chdir(self.curdir)
        inp_path = "Train/Preprocessing/EDA"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir

    def ner_inputs_file(self):
        os.chdir(self.curdir)
        inp_path = "Train/Preprocessing/Manual_annotation_NER"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir
    
    def clean_text_inputs(self):
        os.chdir(self.curdir)
        inp_path = "Train/Preprocessing/Clean_text"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir

    def ner_model(self):
        os.chdir(self.curdir)
        inp_path = "Train/NER"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir

    def nel_kb_inputs(self):
        os.chdir(self.curdir)
        inp_path = "Train/NEL"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir

    def nel_kb_vocab(self):
        os.chdir(self.curdir)
        inp_path = "Train/NEL/NEL_Model"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir

    def nel_wo_training(self):
        os.chdir(self.curdir)
        inp_path = "Train/NEL/NEL_Without_Training"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir

    def nel_with_training(self):
        os.chdir(self.curdir)
        inp_path = "Train/NEL/NEL_With_Training"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir

    def ml_model(self):
        os.chdir(self.curdir)
        inp_path = "Train/ML"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir

    def final_output(self):
        os.chdir(self.curdir)
        inp_path = "Test/final_output"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir
    
    def read_datafolder(self):
        os.chdir(self.curdir)
        inp_path = "Data/"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir

    def read_preinput(self):
        os.chdir(self.curdir)
        inp_path = "Pre_loaded_inputs_and_outputs/"
        if not os.path.exists(inp_path):
            os.makedirs(inp_path)
        else:
            pass
        os.chdir(inp_path)
        return self.curdir
    
    def reset_directory(self):
        os.chdir(self.reset)

    def read_dir_files(self, ext, function):
        try:
            files = []
            thisdir = os.getcwd()
            print(thisdir)
            for _, _, f in os.walk(thisdir):
                for file in f:
                    if ext!=None:
                        if file.endswith(ext): files.append(file)
                    else: files.append(file)
            print(f"We found below files/models of {function} folder.")
            print(files)
            inp = input("Please provide the name to choose the file. If you want to choose other path enter 'y':\n")
            if inp.lower() == 'y':
                return None
            if inp in files: return inp
            else:
                inp = input("Choose the correct file and try again:\n")
                if inp in files: return inp
                else:
                    print("File not found.We are sorry...")
                    exit
        except FileExistsError:
            return None
    
    def ner_folder(self):
        try:
            dirs=[]
            thisdir = os.getcwd()
            for _, d, _ in os.walk(thisdir):
                # for dir in d:
                dirs.append(d)
            print(f"We found below models in the existing trained NER folder.")
            if dirs!=[]:print(dirs[0])
            inp = input("Please provide the folder name to choose for training. If you want to choose in other path enter 'y':\n")
            if inp.lower() == 'y':
                return None
            if dirs and inp in dirs[0]: return inp
            else:
                inp = input("Choose the correct file and try again:\n")
                if dirs and inp in dirs[0]: return inp
                else:
                    print("File not found.We are sorry...")
                    exit
        except FileExistsError:
            return None