import pandas as pd
import spacy
from autocorrect import Speller
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score , f1_score
from sklearn.decomposition import LatentDirichletAllocation as lda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns

class EvaluateML:
    def __init__(self, df, clusters):
        self.df = df
        self.clusters = clusters
        self.intrensic_report()

    def prepare_count_vectorizer(self, corpus):
        count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
        X = count_vectorizer.fit_transform(corpus)
        return ( X , count_vectorizer)


    def prepare_tfidf(self, corpus):
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(corpus)
        return ( X , tfidf)


    def fit_kemeans(self, no_of_clusters , X , vectorizer):
        kmeans = KMeans(n_clusters=no_of_clusters).fit(X)
        labels = kmeans.labels_
        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
        return ( {'SSE' : kmeans.inertia_ , 'Silhouette' : silhouette_score})

    # Generating intensic report with elbow method & silhouette co-efficient values
    def intrensic_report(self):

        # Preparing corpus with cleaned data
        corpus = []
        for sentence in self.df['clean_text']:
            sentence = sentence.lstrip()
            corpus.append(sentence)

        # Preparing Counter vectorizer matrix
        X_bow , count_vectorizer = self.prepare_count_vectorizer(corpus)
        # Preparing TF-IDF matrix
        X_tfidf , tfidf = self.prepare_tfidf(corpus)

        # Generating SSE score and Silhouette score at each number of clusters from 2 to 15
        sse_bow = {}
        silh_score_bow = {}
        sse_tfidf = {}
        silh_score_tfidf = {}
        for k in range (2 , 16):
            score = self.fit_kemeans(k , X_bow , count_vectorizer)
            sse_bow [k] = score['SSE']
            silh_score_bow[k] = score['Silhouette']

            score = self.fit_kemeans(k , X_tfidf , tfidf)
            sse_tfidf [k] = score['SSE']
            silh_score_tfidf[k] = score['Silhouette']  

        # Preparing report and saving report as CSV file
        Report_df = pd.DataFrame(columns = [' ' ,' ' , 'WSS_BOW' , 'WSS_TFIDF' , ' ' , ' ' , 'silhouette_BOW' , 'silhouette_TFIDF'])
        Report_df['WSS_BOW'] = pd.Series(sse_bow)
        Report_df['WSS_TFIDF'] = pd.Series(sse_tfidf)
        Report_df['silhouette_BOW'] = pd.Series(silh_score_bow)
        Report_df['silhouette_TFIDF'] = pd.Series(silh_score_tfidf)
        Report_df.to_csv('Evaluation_Reports/internal/Intrinsic_Evaluation_Report.csv')

        # Preparing Plot and saving plot as png file
        fig , ax = plt.subplots ( nrows = 2 , ncols = 2 , figsize = (25,15))

        for i in [1,2,3,4]:
            plt.subplot(2,2,i)
            if i == 1: dic , label , col = sse_bow , "WSS - BOW" , 'r'
            elif i == 2: dic , label , col = sse_tfidf , "WSS - TFIDF" , 'b'
            elif i == 3: dic , label , col = silh_score_bow , "silhouette - BOW" , 'g'
            elif i == 4: dic , label , col = silh_score_tfidf , "silhouette - TFIDF" , 'k'
            plt.plot(list(dic.keys()), list(dic.values()) , color = col ,  marker='o',linewidth=2, markersize=12)
            plt.xlabel("Number of cluster" ,  fontsize=10)
            plt.ylabel(label.split('-')[0],  fontsize=10)
            plt.title(label, fontsize=20)

        plt.savefig('Evaluation_Reports/internal/Intrinsic_Evaluation_Report.png')
        plt.savefig('Evaluation_Reports/static/Intrinsic_Evaluation_Report.png')
        self.extrensic_report(X_bow , X_tfidf)

    def train_topic_model(self, docs , n_clusters):
        lda_random_state = 100
        lda_n_iter = 100
        n_top_words = 20
        print("Topic modeling using LDA...")
        d2w_vect = TfidfVectorizer(stop_words='english', max_df=0.30)
        d2w = d2w_vect.fit_transform(docs)
        model = lda(n_components=n_clusters, max_iter=lda_n_iter, random_state=lda_random_state)
        model.fit(d2w)
        print("\nTopical words:")
        print("-" * 20)
        words = [w for w, i in d2w_vect.vocabulary_.items()]
        for i, topic_dist in enumerate(model.components_):
            top_word_ids = np.argsort(topic_dist)[:-n_top_words:-1]
            topic_words = [words[id_]
                        for id_ in top_word_ids]
            print('Topic {}: {}'.format(i, ', '.join(topic_words)))
        topic_values = model.fit_transform(d2w)
        return topic_values

    def extrensic_report(self, X_bow , X_tfidf):

        # Generating Labels with LDA
        topic_values = self.train_topic_model(self.df['clean_text'] , self.clusters)

        # Stroing all labels in Extensic report Data frame
        Extr_report_df = pd.DataFrame(columns =['LDA_labels' , 
                                            'BOW_Manhattan_labels',
                                            'BOW_Euclidian_labels',
                                            'BOW_Minkowski_labels',
                                            'BOW_Cosine_labels',
                                            'TFIDF_Manhattan_labels',
                                            'TFIDF_Euclidian_labels',
                                            'TFIDF_Minkowski_labels',
                                            'TFIDF_Cosine_labels'
                                            ]
                                    )
        
        Extr_report_df['LDA_labels'] = topic_values.argmax(axis=1)

        # Preparing differnt columns for each K-means experimental setup
        col_names = Extr_report_df.columns[1:]

        # Generating labels and stroing lables for each K-means experimental setup
        i = 0
        for X in [X_bow , X_tfidf]:
            for d in ['cityblock' , 'euclidean' , 'manhattan' , 'cosine']:
                kmedoids = KMedoids(self.clusters, random_state=0,metric=d, init='k-medoids++').fit(X)
                Extr_report_df[col_names[i]] = kmedoids.labels_
                i = i+1

        # Plotting confusion matrix , accuracy and F1-score for each K-means experimental setup
        fig , ax = plt.subplots ( nrows = 2 , ncols = 4 , figsize = (20,8))
        col_list = ["YlGnBu" , "BrBG" , "PRGn" ,   "Spectral" , "PuOr" , "RdYlGn" , "PiYG" , "RdBu"]
        i = 0
        for col_name in col_names:
            cm = confusion_matrix(Extr_report_df['LDA_labels'], Extr_report_df[col_name])
            acc = accuracy_score(Extr_report_df['LDA_labels'], Extr_report_df[col_name])
            f1_sc = f1_score(Extr_report_df['LDA_labels'], Extr_report_df[col_name], average='macro')
            sns.heatmap(cm, annot=True, ax = ax.flat[i] , cbar = False ,cmap=col_list[i])
            ax.flat[i].text(x=0.65, y=-0.3 , s=col_name, fontsize=12, weight='bold')
            ax.flat[i].text(x=0.65, y=-0.1, s='accuracy:' + str(round(acc,2)*100) + ' %         f1_score:' + str(round(f1_sc,2)*100) + ' %' , fontsize=8, alpha=0.75)
            i = i+1

        fig.text(0.5, 0.04, 'Predicted labels', ha='center', va='center')
        fig.text(0.06, 0.5, 'True labels', ha='center', va='center', rotation='vertical')

        # Saving the plot and report into hard disk
        plt.savefig("Evaluation_Reports/external/Extrinic_Evaluation_Report.png")
        plt.savefig("Evaluation_Reports/static/Extrinic_Evaluation_Report.png")
        Extr_report_df.to_csv('Evaluation_Reports/external/Extrinic_Evaluation_Report.csv')
        print("Evaluation reports are saved in Evaluation_Reports folder")