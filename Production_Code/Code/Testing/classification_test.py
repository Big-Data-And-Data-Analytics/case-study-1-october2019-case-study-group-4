import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def test_cluster_model(model,docs):
    email_df = docs
    if model != None: filename = model
    else:filename = input("Please enter the classification model location:\n")
    try:
        kmeans = pickle.load(open(filename, 'rb'))
    except FileNotFoundError:
        print("Model not found in the directory")
        exit
    email_df.dropna(subset=['clean_text'], inplace=True)
    vect = TfidfVectorizer(stop_words='english', max_df=0.30)
    vectorizer = vect.fit_transform(email_df['clean_text'])
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(vectorizer.toarray())
    prediction = kmeans.predict(Z)
    email_df['cluster'] = prediction
    return email_df