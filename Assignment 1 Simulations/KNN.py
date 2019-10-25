from string import punctuation
from os import listdir
from numpy import array
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import numpy as np

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv1'):
			continue
		if not is_trian and not filename.startswith('cv1'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, True)
negative_lines = process_docs('txt_sentoken/neg', vocab, True)

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
docs = negative_lines + positive_lines
tokenizer.fit_on_texts(docs)

# encode training data set
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
print(Xtrain.shape)

# load all test reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, False)
negative_lines = process_docs('txt_sentoken/neg', vocab, False)
docs = negative_lines + positive_lines
# encode training data set
Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
print(Xtest.shape)

# encode training data set
# Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# load all test reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, False)
negative_lines = process_docs('txt_sentoken/neg', vocab, False)
docs = negative_lines + positive_lines
# encode training data set
Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

print(ytrain.shape)
print(ytest.shape)

n_words = Xtest.shape[1]

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score


k_range = [1,3,5]
scores={}
scores_list = []
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(Xtrain,ytrain)
	y_pred = knn.predict(Xtest)
	scores[k]= metrics.accuracy_score(ytest,y_pred)
	scores_list.append(metrics.accuracy_score(ytest,y_pred))
	score = cross_val_score(knn, Xtrain, ytrain, cv=5)
print(score.mean())

print("for 2 norm k=1,3,5: ")
print(scores_list)

scores={}
scores_list = []
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k,metric='manhattan')
	knn.fit(Xtrain,ytrain)
	y_pred = knn.predict(Xtest)
	scores[k]= metrics.accuracy_score(ytest,y_pred)
	scores_list.append(metrics.accuracy_score(ytest,y_pred))
	score = cross_val_score(knn, Xtrain, ytrain, cv=5)
print(score.mean())

print("for 1 norm k=1,3,5: ")
print(scores_list)

scores={}
scores_list = []
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k,metric='chebyshev')
	knn.fit(Xtrain,ytrain)
	y_pred = knn.predict(Xtest)
	scores[k]= metrics.accuracy_score(ytest,y_pred)
	scores_list.append(metrics.accuracy_score(ytest,y_pred))
	score = cross_val_score(knn, Xtrain, ytrain, cv=5)
print(score.mean())


print("for infinity norm k=1,3,5: ")
print(scores_list)