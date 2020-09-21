import spacy
import pandas as pd


def named_entity_recognition(text):
    email_df = text
    person = []
    product = []
    time = []
    date =[]
    location = []
    organization = []
    for ent in email_df.ents:
        if ent.label_ == "Person":
            person.append((ent.text,ent.kb_id_))
        elif ent.label_ == "Product":
            product.append([ent.text,ent.kb_id_])
        elif ent.label_ == "TIME":
            time.append([ent.text,ent.kb_id_])
        elif ent.label_ == "Date":
            date.append([ent.text,ent.kb_id_])
        elif ent.label_ == "Location":
            location.append([ent.text,ent.kb_id_])
        elif ent.label_ == "Organization":
            organization.append([ent.text,ent.kb_id_])
#    email_df['person'] = person
    dict = {'Person': person, 'Product': product, 'Time': time, 'Date': date, 'Location': location, 'Organization': organization }
    return dict

def ner_df(model, dataframe):
    email_df = dataframe
    nlp = spacy.load(model)
    email_df['Conent_NER'] = email_df['email_body'].apply(lambda x: nlp(x))
    email_df['NER_Info'] = email_df['Conent_NER'].apply(named_entity_recognition)
    return email_df
