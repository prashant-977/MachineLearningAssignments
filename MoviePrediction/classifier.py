import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import numpy as np

data_from_csv = pd.read_csv('movie_train.csv', header=0)
data_from_csv['Runtime'] = data_from_csv['Runtime'].apply(str)

def _generate_data(data):
	'''Generates each line of movie at a time'''
	for item in data:
		yield item


# Split the data into testing data and training data 
training_data = _generate_data(data_from_csv[['Runtime', 'Director', 'Writer', 'Actor']][:-15].values)
testing_data = _generate_data(data_from_csv[['Runtime', 'Director', 'Writer', 'Actor']][-15:].values)

#Generate features and transform using countVectorizer and TfidfTransformer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(str(x) for x in training_data)

#transform the data into numpy array that .fit expects
tf_transformer = TfidfTransformer()
X_train_tfidf = tf_transformer.fit_transform(X_train_counts)

#NaiveBayes classifier
clf = MultinomialNB().fit(X_train_tfidf, data_from_csv['Target'][:-15].values)

#now test on 2016 data 
test_data_from_csv = pd.read_csv('movie_test.csv', header=0)
def _generate_test_data():
	'''generates each row of csv file at a time'''
	for item in test_data_from_csv[['Runtime', 'Director', 'Writer', 'Actor']].values:
		yield item
test_gen = _generate_test_data()


#transform the string data to numpy array
X_c = count_vect.transform(str(movie) for movie in test_gen)
X_tf = tf_transformer.transform(X_c)
for movie, target in zip(test_data_from_csv['Name'], clf.predict(X_tf)):
	if target == 1:
		print ('{movie} is Success'.format(movie=movie))
	else:
		print ('{movie} is flop'.format(movie=movie))
print () 
print ('Now svm implementation')
print ()

# SVM classifier with deault parameters
svm_clf = SGDClassifier(loss='hinge')
svm_clf.fit(X_train_tfidf, data_from_csv['Target'][:-15].values)
for movie, target in zip(test_data_from_csv['Name'], svm_clf.predict(X_tf)):
	if target == 1:
		print ('{movie} is Success'.format(movie=movie))
	else:
		print ('{movie} is flop'.format(movie=movie))


#evaluation of performance
test_counts = count_vect.transform(str(movie) for movie in testing_data)
test_tfidf = tf_transformer.transform(test_counts)
nv_predicted = clf.predict(test_tfidf)
print ('Naive Bayes accuracy ', np.mean(nv_predicted == data_from_csv['Target'][-15:].values))

svm_predicted = svm_clf.predict(test_tfidf)
print ('SVM accuracy ',np.mean(svm_predicted == data_from_csv['Target'][-15:].values))

def main():
	pass

if __name__ == '__main__':
	main()