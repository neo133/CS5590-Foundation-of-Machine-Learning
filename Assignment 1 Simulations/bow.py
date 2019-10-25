# Python3 code for preprocessing text 
import nltk 
import re 
import numpy as np 

def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# execute the text here as : 
# text = """ # place text here """ 
filename = 'negative.txt'
text = load_doc(filename)
# text = lines.readlines()
# print(text) 
dataset = nltk.sent_tokenize(text) 
for i in range(len(dataset)): 
	dataset[i] = dataset[i].lower() 
	dataset[i] = re.sub(r'\W', ' ', dataset[i]) 
	dataset[i] = re.sub(r'\s+', ' ', dataset[i]) 

# Creating the Bag of Words model 
word2count = {} 
for data in dataset: 
	words = nltk.word_tokenize(data) 
	for word in words: 
		if word not in word2count.keys(): 
			word2count[word] = 1
		else: 
			word2count[word] += 1

import heapq 
freq_words = heapq.nlargest(100, word2count, key=word2count.get)

X = [] 
for data in dataset: 
	vector = [] 
	for word in freq_words: 
		if word in nltk.word_tokenize(data): 
			vector.append(1) 
		else: 
			vector.append(0) 
	X.append(vector) 
X = np.asarray(X) 
