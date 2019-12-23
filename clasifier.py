# coding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
import numpy as np

def load_dataset(type):
    container_path = "documents/" + type + "_dataset"
    dataset = load_files(container_path, description=None,
        categories=None, load_content=True, shuffle=True, encoding="utf-8", decode_error='strict', random_state=0)
    num_examples = len(dataset.data)
    print("Loaded " + type + " dataset with " + str(num_examples) + " examples")
    return dataset

# Builds the naive bayes classifier
def naive_bayes_clasifier():
    clasifier = Pipeline([('vectorizer', CountVectorizer()),
                                ('transformer', TfidfTransformer()),
                                ('nb_clasifier', MultinomialNB())])
    return clasifier

# Builds the SVM classifier
def svm_clasifier():
    clasifier = Pipeline([('vectorizer', CountVectorizer()),
                            ('transformer', TfidfTransformer()),
                            ('svm_clasifier', SGDClassifier(loss='hinge',
                                penalty='l2',alpha=1e-3, max_iter=20, random_state=42))])
    return clasifier

# Train a clasifier with the training data
def train_clasifier(clasifier, training_data):
    return clasifier.fit(training_data.data, training_data.target)

# Predicts the average performance of the clasifier
def calculate_prediction(clasifier, test_data):
    predicted = clasifier.predict(test_data.data)
    mean_prediction = np.mean(predicted == test_data.target)
    return mean_prediction

def build_and_test_clasifier(clasifier_type, training_data, test_data):
    print("\nBuilding and testing a " + clasifier_type + " clasifier")
    if clasifier_type == "Naive Bayes":
        clasifier = naive_bayes_clasifier()
    elif clasifier_type == "SVM":
        clasifier = svm_clasifier()
    clasifier = train_clasifier(clasifier,training_data)
    prediction = calculate_prediction(clasifier, test_data)
    print("Performance prediction: " + str(prediction))
    return clasifier


training_data = load_dataset("train")
test_data = load_dataset("test")

nb_clasifier = build_and_test_clasifier("Naive Bayes",  training_data, test_data)
svm_clasifier = build_and_test_clasifier("SVM",  training_data, test_data)
