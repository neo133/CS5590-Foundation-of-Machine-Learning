from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA, FastICA

categories = None

newsgroups_train = fetch_20newsgroups(subset='train')
labels = newsgroups_train.target

true_k = 12

#instantiate CountVectorizer()
cv=CountVectorizer()
 
# this steps generates word counts for the words in your docs
# word_count_vector=cv.fit_transform(newsgroups_train.data)

# tfidf_transformer=TfidfTransformer(norm ='l2',smooth_idf=True,use_idf=True)
# X = tfidf_transformer.fit(word_count_vector)



#vectorize

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', sublinear_tf=False)


X = vectorizer.fit_transform(newsgroups_train.data)
X = X.toarray()
#clustering
# km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
# km = SpectralClustering(n_clusters=20, gamma=0.01, affinity='rbf', assign_labels='kmeans')

# km.fit(X)
# transformer = FactorAnalysis(n_components=10, random_state=0)
# f = transformer.fit_transform(X)

pca = PCA()
S_pca_ = pca.fit(X).transform(X)

ica = FastICA(n_components=5)
S_ica_ = ica.fit(X).transform(X)  # Estimate the sources

S_ica_ /= S_ica_.std(axis=0)


# order_centroids = km.cluster_centers_.argsort()[:, ::-1]

# terms = vectorizer.get_feature_names()

# for i in range(true_k):
#     print("cluster %d:" % i)
#     for ind in order_centroids[i,:20]:
#         print('%s' % terms[ind])
#     print()


#Performance
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, S_ica_.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, gmm.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, gmm.labels_))
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, gmm.labels_, sample_size=1000))
# print("Ajusted rand score: %0.3f" % metrics.adjusted_rand_score(labels, gmm.labels_))
# print("AMI: %0.3f" % metrics.adjusted_mutual_info_score(labels, gmm.labels_))
# print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels, gmm.labels_))
# print("FMI: %0.3f" % metrics.fowlkes_mallows_score(labels, gmm.labels_))
