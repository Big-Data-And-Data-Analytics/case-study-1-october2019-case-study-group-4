class CleanEmails():
    def __init__(self, dataframe, col):
        import pandas as pd
        import spacy
        from autocorrect import Speller
        self.nlp = spacy.load('en_core_web_lg')
        self.check = Speller(lang='en')
        self.email_df = dataframe
        if 'email_body' in self.email_df.columns: self.email_df['tokens'] = self.email_df['email_body'].apply(self.nlp)
        else: self.email_df['tokens'] = self.email_df[col].apply(self.nlp)

    # Capturing the standard entities from the emails
    def entity_list(self, data):
        entities = []
        ent_text = []
        for ent in data.ents:
            entities.append((ent.text, ent.start, ent.end-1,ent.label_))
            ent_text.append(ent.text)
        return ent_text, entities

    # Removing email chains using spacy lg model based on the token shape and position from the emails to keep original email along with fully cleaned email for machine learning purpose
    def clean_content(self, x):
        prev_token = ''
        clean_tokens_text = []
        entity_text = []
        tokens = ''
        end_index = ''
        ent_text = self.entity_list(x)
        entity_text.append(ent_text)
        # entities.append(entity)
        for token in x:
            if token.text == 'Forwarded' or token.text == 'From':
                if end_index == '':
                    end_index = token.text
            elif token.like_email==True or str(token.shape_).find('@')>0:
                prev_token = token.text
            elif prev_token != '' and token.shape_ == 'dd/dd/dddd':
                if end_index == '':
                    end_index = prev_token
            if end_index == '' and token.is_alpha == True and len(token.lemma_) > 2 and token.is_stop == False and token.text not in ent_text:
                tokens = token.lemma_.lower()
                clean_tokens_text.append(tokens)
        clean_data = (' ').join(x for x in clean_tokens_text)
        clean_data = self.check(clean_data)
        return clean_data

    def clean_emails(self):
        self.email_df['clean_text'] = self.email_df['tokens'].apply(self.clean_content)
        return self.email_df
