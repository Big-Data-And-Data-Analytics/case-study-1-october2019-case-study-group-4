import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
# from Evaluate_ML import *

class TrainML:
    def __init__(self, df,inp_data):
        self.df = df
        self.inp_data = inp_data

    # To find the elbow strength based on cluster intertia and destortion score with relative strength
    def cluster_strength(self, score):
        delta1 = []
        delta2 = []
        strength = []
        rel_strength = []
        j = 0
        max_num = 0
        if score:cluster_f,_ = score[0]
        else: cluster_f = 2
        for i in range(len(score)): # Finding the relative strength between the preceding score and the succeding score
            if i > j:
                _,score1 = score[i-1]
                _,score2 = score[i]
                delta1.append(score1 - score2)
                if i > j+1: delta2.append(delta1[i-1] - delta1[i])
                else: delta2.append(0)
                strgt = delta2[i] - delta1[i]
                if strgt > 0: strength.append(strgt)
                else: strength.append(0)
                if strength[i-1] and strength[i-1] > 0:
                    cluster,_ = score[i-1]
                    rel_strgt = strength[i-1]/cluster
                    rel_strength.append(rel_strgt)
                    if max_num < rel_strgt:
                        max_num = max(rel_strength)
                        cluster_f = cluster
            else:
                delta1.append(0)
                delta2.append(0)
        return cluster_f, max_num

    # Train the KMeans clustering with optimal clusters based on above cluster strength function
    def train_ml(self, n_random_state = 42,n_iter = 100, n_top_words = 20):
        self.df.dropna(subset=[self.inp_data], inplace=True)
        docs = self.df[self.inp_data]
        print("Clustering using KMeans...")
        d2w_vect = TfidfVectorizer(stop_words='english', max_df=0.30)
        d2w = d2w_vect.fit_transform(docs)
        vect_norm = normalize(d2w)
        vect_array = vect_norm.toarray()
        pca = PCA(n_components = 2)
        Y = pca.fit_transform(vect_array)
        max_clusters = int(len(d2w_vect.get_feature_names())*0.25/100)
        # print(max_clusters)
        # print(len(self.df))
        distortions = []
        inertias = []
        print("Please wait for model training..........")
        for i in range(1,max_clusters): # finding the optimal number of cluster by iterating KMeans and calculating the cluster strength
            kmeanModel = KMeans(n_clusters=i, init='k-means++', max_iter=n_iter, algorithm = 'auto', random_state=n_random_state)
            kmeanModel.fit(vect_array)  
            distortions.append((i, sum(np.min(cdist(vect_array, kmeanModel.cluster_centers_, 
                            'euclidean'),axis=1)) / vect_array.shape[0])) # finding the distrortion based on cluster centers using euclidean distance measure
            inertias.append((i, kmeanModel.inertia_))
        # print(inertias)
        # print(distortions)
        cluster_in, _ = self.cluster_strength(inertias)
        cluster_de, _ = self.cluster_strength(distortions)
        max_clusters = cluster_in
        print("clusters are", max_clusters)
        kmeans = KMeans(n_clusters=max_clusters, init='k-means++', max_iter=n_iter, algorithm = 'auto', random_state=n_random_state)
        kmeans.fit(Y)
        filename = 'kmeans_model.sav'
        pickle.dump(kmeans, open(filename, 'wb'))

        kmeans = pickle.load(open(filename, 'rb'))
        vect = TfidfVectorizer(stop_words='english', max_df=0.30)
        vectorizer = vect.fit_transform(docs)
        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(vectorizer.toarray())
        prediction = kmeans.predict(Z)
        # print(prediction)
        # eval_model = EvaluateML(self.df, max_clusters)
        print("Clustering model has been successfully trained")

