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
    dataset = load_files(container_path, shuffle=True, encoding="utf-8")
    num_examples = len(dataset.data)
    print("Loaded " + type + " dataset with " + str(num_examples) + " examples")
    return dataset

# Builds the naive bayes classifier
def naive_bayes_classifier():
    classifier = Pipeline([('vectorizer', CountVectorizer()),
                                ('transformer', TfidfTransformer()),
                                ('nb_classifier', MultinomialNB())])
    return classifier

# Builds the SVM classifier
def svm_classifier():
    classifier = Pipeline([('vectorizer', CountVectorizer()),
                            ('transformer', TfidfTransformer()),
                            ('svm_classifier', SGDClassifier(loss='hinge',
                                penalty='l2',alpha=1e-3, max_iter=20, random_state=42))])
    return classifier

# Train a classifier with the training data
def train_classifier(classifier, training_data):
    return classifier.fit(training_data.data, training_data.target)

# Predicts the average performance of the classifier
def calculate_prediction(classifier, test_data):
    predicted = classifier.predict(test_data.data)
    mean_prediction = np.mean(predicted == test_data.target)
    return mean_prediction

def build_and_test_classifier(classifier_type, training_data, test_data):
    print("\nBuilding and testing a " + classifier_type + " classifier")
    if classifier_type == "Naive Bayes":
        classifier = naive_bayes_classifier()
    elif classifier_type == "SVM":
        classifier = svm_classifier()
    classifier = train_classifier(classifier,training_data)
    prediction = calculate_prediction(classifier, test_data)
    print("Performance prediction: " + str(prediction))
    return classifier


training_data = load_dataset("train")
test_data = load_dataset("test")

nb_classifier = build_and_test_classifier("Naive Bayes",  training_data, test_data)
svm_classifier = build_and_test_classifier("SVM",  training_data, test_data)
