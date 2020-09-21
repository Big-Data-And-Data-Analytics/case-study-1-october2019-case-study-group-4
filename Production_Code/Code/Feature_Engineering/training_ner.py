import spacy
import random
import spacy.cli
import pickle
import pandas as pd
import email
from spacy.util import minibatch, compounding
from spacy.pipeline import Tagger
from spacy.pipeline import DependencyParser


#Function to train the model with NER pipeline
def train_spacy(data,iterations,model=None):
    TRAIN_DATA = pickle.load(open(data,'rb'))
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    
    if model is None:
        optimizer = nlp.begin_training()
    else:
        print("Existing entities in the model are:",move_names)
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                text, annotations = zip(*batch)
                nlp.update(
                    text,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    
    #custom_ner_model = spacy.load(nlp)
    
    nlp_core_model = spacy.load("en_core_web_lg")
    
    tagger = Tagger(nlp_core_model.vocab)
    nlp.add_pipe(tagger, before="ner")

    parser = DependencyParser(nlp_core_model.vocab)
    nlp.add_pipe(parser, before="ner")
    nlp.begin_training()
    return TRAIN_DATA, nlp
