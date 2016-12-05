#importing modules
#/usr/bin/python
from flask import Flask, jsonify
import nltk
from nltk.corpus import names
import random
import cPickle
from collections import Counter


class GenderPredictor(object):
	def __init__(self):
		pass

	#extracting some features
	def name_features(self, word):
		return {'last_letter': word[-1],
				'last_two' : word[-2:],
				'last_is_vowel' : (word[-1] in 'aeiou')}

	def gender_features(self, word= None, train=True):
		maleNames=[name for name in names.words('male.txt')]
		femaleNames = [name for name in names.words('female.txt')]
		if train:
			featureset = list()
			for name in maleNames:
				features = self.name_features(name)
				male_prob, female_prob = self.get_prob(name, maleNames)
				features['male_prob'] = male_prob
				features['female_prob'] = female_prob
				featureset.append((features,'male'))
			for name in femaleNames:
				features = self.name_features(name)
				male_prob, female_prob = self.get_prob(name, femaleNames)
				features['male_prob'] = male_prob
				features['female_prob'] = female_prob
				featureset.append((features,'female'))
		else:
			features = self.name_features(word)
			male_prob, female_prob = self.get_prob(word, list(maleNames+femaleNames))
			features['male_prob'] = male_prob
			features['female_prob'] = female_prob
			featureset = features
		return featureset



	def get_prob(self, nameTuple, list):
		counts = Counter(list)
		male_prob = (counts[nameTuple] * 1.0) / (sum(counts.values()))
		if male_prob == 1.0:
			male_prob = 0.99
		elif male_prob == 0.0:
			male_prob = 0.01
		else:
			pass
		female_prob = 1.0 - male_prob
		return (male_prob, female_prob)


	def train(self, filename, new=True, train_size = 0.7):
		if new:
			featuresets = self.gender_features()
			random.shuffle(featuresets)
			split = int(train_size * len(featuresets))
			print split, len(featuresets)
			train_set, test_set = featuresets[:split], featuresets[split:]
			classifier = nltk.NaiveBayesClassifier.train(train_set)
			with open(filename, "wb") as f:
				cPickle.dump(classifier, f)
		else:
			with open(filename, "rb") as f:
				classifier = cPickle.load(f)
		return classifier


	def predict(self, classifier, data):
		return classifier.classify(self.gender_features(data.lower(), False))


app = Flask(__name__)

@app.route('/')
def index():
	return "Lets start predicting!"


@app.route('/API/<name>', methods=['GET'])
def get_gender(name):
	modelObj = GenderPredictor()
	clf = modelObj.train('NaiveBayesClassifier.pkl', False)
	gender = modelObj.predict(clf, name.lower())
	return jsonify({'name': name, 'gender': gender})

if __name__ == '__main__':
	app.run(debug=True)
	#Uncomment when you want to rebuild the model
	#modelObj = GenderPredictor()
	#clf = modelObj.train('NaiveBayesClassifier.pkl')
