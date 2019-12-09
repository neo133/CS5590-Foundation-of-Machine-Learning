# -*- coding: utf-8 -*-
"""
Created on Sun Dec 08 10:11:59 2019

@author: Francis
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing


newsgroups_train = fetch_20newsgroups(subset='train')
labels = newsgroups_train.target

vectorizer = TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')
X = vectorizer.fit_transform(newsgroups_train.data)
Y = preprocessing.normalize(X, norm='l1', axis=1, copy=True, return_norm=False)

#--------------Kernalize K-means------------------------------------------
km=SpectralClustering(n_clusters=20,gamma= 0.01, affinity='rbf')
km.fit(Y)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, km.labels_, sample_size=1000))

#Performance
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels, km.labels_))
print("NMI:%0.3f" % metrics.normalized_mutual_info_score(labels,km.labels_))
print("AMI:%0.3f" %metrics.adjusted_mutual_info_score(labels,km.labels_))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("FMI:%0.3f" % metrics.fowlkes_mallows_score(labels, km.labels_))