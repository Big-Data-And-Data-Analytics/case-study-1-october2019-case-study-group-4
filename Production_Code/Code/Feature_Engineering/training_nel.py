import spacy
import pandas as pd
from spacy.kb import KnowledgeBase
from spacy.vocab import Vocab
from spacy.util import minibatch, compounding
import random
import pickle
from ..path_creation import OsPath

folder = OsPath()

class NelTraining():

    def __init__(self,custom_ner_model):
        self.custom_ner_model = spacy.load(custom_ner_model)

    def creating_knowledge(self,ner_data):
        ner_train_data = pickle.load(open(ner_data,'rb'))
        ner_results = []
        for i,j in ner_train_data:
            remove_quotes = i.replace('"', '')
            doc2 = self.custom_ner_model(remove_quotes)
            for ent in doc2.ents:
                ner_results.append((ent.label_,remove_quotes,ent.start_char,ent.end_char,ent.text))
        df_ner_results = pd.DataFrame(ner_results, columns = ['Entity' , 'Text', 'Start', 'End', 'Name'])
        folder.nel_kb_inputs()
        file_name = str(input("Enter the File Name of the KB with the extension .csv :"))
        df_ner_results.to_csv(file_name, index=False, sep = ';')
        df_base = df_ner_results.groupby(['Entity','Name']).size().reset_index(name='Frequency')
        file_name = str(input("Enter the File Name to save the inputs for Training with the extension .csv :"))
        df_base.to_csv(file_name, index=False, sep = ';')


    def settingup_knowledgebase(self,names,train_data_2):
        
        QID = names['QID'].values.tolist()
        Names = names['Names'].values.tolist()
        Frequency = names['Frequency'].values.tolist()
        descript = []
        for desc in names['Description']:
            descript.append(self.custom_ner_model(desc).vector)
            
        print("Setting up entities \n")
            
        kb = KnowledgeBase(vocab=self.custom_ner_model.vocab, entity_vector_length=96)
        kb.set_entities(entity_list=QID, freq_list=Frequency, vector_list=descript)
        
        print("Setting up Alias \n")

        print("\n")
        print("Spacy Pipeline \n")

        print(self.custom_ner_model.pipe_names)

        #kb_dump_file = str(input("Enter the KB Dump name: "))
        #kb_vocab_folder = str(input("Enter the KB Vocab name: "))
        folder.nel_kb_vocab()

        alias_prep =  list(zip(Names, QID))
        folder.nel_kb_vocab()
        for i,j in alias_prep:
            names_alias = str(i)
            list_qid = []
            list_qid.append(j)
            prob = []
            prob.append(int(1.0))
            kb.add_alias(alias=names_alias, entities=list_qid, probabilities=prob)
            
            kb.dump("KB_Dump")
            kb.vocab.to_disk("KB_Vocab")
            
        print("\n")
        print("Knowbase dump and Vocab are stored in a local disk")
        
        train_data_dict_2 = train_data_2.to_dict('records')

        dataset_2 = []
        for data in train_data_dict_2:
            Text = data['Text']
            Name = data['Name']
            QID = data['QID']
            offset = (data["Start"], data["End"])
            links_dict = {QID: 1.0}
            dataset_2.append((Text, {"links": {offset: links_dict}}))

        self.custom_ner_model.vocab.from_disk("KB_Vocab")
        self.custom_ner_model.vocab.vectors.name = "spacy_pretrained_vectors"
        kb = KnowledgeBase(vocab=self.custom_ner_model.vocab)
        kb.load_bulk("KB_Dump")

        TRAIN_DOCS = []
        for text, annotation in dataset_2:
            doc = self.custom_ner_model(text)     # to make this more efficient, you can use nlp.pipe() just once for all the texts
            TRAIN_DOCS.append((doc, annotation))

        print("\n")
        print("Training started for Named Entity Linking \n")

        entity_linker = self.custom_ner_model.create_pipe("entity_linker", config={"incl_prior": False})
        entity_linker.set_kb(kb)
        self.custom_ner_model.add_pipe(entity_linker, last=True)  
        
        other_pipes = [pipe for pipe in self.custom_ner_model.pipe_names if pipe != "entity_linker"]
        with self.custom_ner_model.disable_pipes(*other_pipes):   # train only the entity_linker
            optimizer = self.custom_ner_model.begin_training()
            for itn in range(500):   # 500 iterations takes about a minute to train
                random.shuffle(TRAIN_DOCS)
                batches = minibatch(TRAIN_DOCS, size=compounding(4.0, 32.0, 1.001))  # increasing batch sizes
                losses = {}
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.custom_ner_model.update(
                        texts,  
                        annotations,   
                        drop=0.2,      # prevent overfitting
                        losses=losses,
                        sgd=optimizer,
                    )
                if itn % 50 == 0:
                    print(itn, "Losses", losses)   # print the training loss
        print(itn, "Losses", losses)
        print("\n")
        print("Spacy Pipeline \n")
        print(self.custom_ner_model.pipe_names)
        ner_dump_name = str(input("Enter the Model name: "))
        
        self.custom_ner_model.to_disk(ner_dump_name)
        
        return self.custom_ner_model
        