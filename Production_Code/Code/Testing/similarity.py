import pandas as pd

class Activity:
    def __init__(self, inp_file):
        # self.inp_file = input("Enter the Phrases CSV file:")
        self.inp_file = inp_file
        try:
            self.df_todo = pd.read_csv(self.inp_file, engine='python')
        except FileNotFoundError:
            print("Entered file doesnot exist in the directory")
            exit

    # Finding the cosine similarity score between the email document and generic email phrases
    def doc_similarity(self, docs, phrases):
        list1 =[]
        list2 =[]
        docs_set = {word for word in docs}
        phrases_set = {word for word in phrases}
        rel_vector = docs_set.union(phrases_set)
        for word in rel_vector: 
            if word in docs_set: list1.append(1)
            else: list1.append(0)
            if word in phrases_set: list2.append(1)
            else: list2.append(0)
        c = 0
        for i in range(len(rel_vector)):
                c+= list1[i]*list2[i]
        base = float((sum(list1)*sum(list2))**0.5)
        if base !=0:
            score = c / base
        else: score = 0
        return score

    # Reading the cosine similarity score and assign the respective label to the email based on the maximum score
    def activity_label(self, x):
        score_f = []
        max_score = 0
        label_i = ''
        for index, row in self.df_todo.iterrows():
            #score = docs.similarity(row['Phrase_tokens'])
            score = self.doc_similarity(x,row['Phrase'])
            score_f.append(score)
            if max_score < score:
                max_score = max(score_f)
                if score > 0.50:
                    label_i = row['Label']
                else:
                    label_i = 'N.A'
        return label_i
    
    def activity_entity(self, df):
        df['activity'] = df['clean_text'].apply(self.activity_label)
        return df