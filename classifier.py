# coding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
import numpy as np

def config_path(parameter):
    config_file = open("config.txt")
    config_data = config_file.read().split("\n")
    if parameter == "train":
        return config_data[0]
    elif parameter == "test":
        return config_data[1]
    elif parameter == "glossary":
        return config_data[2]

def load_dataset(type):
    container_path = config_path(type)
    dataset = load_files(container_path, shuffle=True, encoding="utf-8")
    num_examples = len(dataset.data)
    print("Loaded " + type + " dataset with " + str(num_examples) + " examples")
    return dataset

def load_glossary():
    glossary_path = config_path("glossary")
    glossary_file = open(glossary_path)
    glossary = glossary_file.read().split("\n")
    glossary_file.close
    return glossary

def build_vectorizer(bigram):
    glossary = load_glossary()
    if bigram:
        vectorizer = CountVectorizer(vocabulary = glossary, ngram_range=(1, 2))
    else:
        vectorizer = CountVectorizer(vocabulary = glossary)
    return vectorizer

# Builds the naive bayes classifier
def naive_bayes_classifier(vectorizer):
    classifier = Pipeline([('vectorizer', vectorizer),
                                ('transformer', TfidfTransformer()),
                                ('nb_classifier', MultinomialNB())])
    return classifier

# Builds the SVM classifier
def svm_classifier(vectorizer):
    classifier = Pipeline([('vectorizer', vectorizer),
                            ('transformer', TfidfTransformer()),
                            ('svm_classifier', SGDClassifier(loss='hinge',
                                penalty='l2',alpha=1e-3, max_iter=20, random_state=42))])
    return classifier

# Builds the type of classifier especified by the parameters
def build_classifier(classifier_type, bigram):
    if bigram:
        print("Building bigram " + classifier_type + " classifier")
    else:
        print("Building unigram " + classifier_type + " classifier")
    vectorizer = build_vectorizer(bigram)
    if classifier_type == "Naive Bayes":
        classifier = naive_bayes_classifier(vectorizer)
    elif classifier_type == "SVM":
        classifier = svm_classifier(vectorizer)
    return classifier

def build_and_train_classifier(classifier_type, training_data, bigram):
    classifier = build_classifier(classifier_type, bigram)
    # Trains the classifier with the training data
    classifier.fit(training_data.data, training_data.target)
    return classifier

def build_and_train_all_classifiers(training_data):
    classifiers = {"Naive Bayes": [], "SVM": [] }
    for classifier_type in classifiers:
        # Building and training unigram classifier
        unigram_classifier = build_and_train_classifier(classifier_type, training_data, bigram = False)
        classifiers[classifier_type].append(unigram_classifier)
        # Building and training bigram classifier
        bigram_classifier = build_and_train_classifier(classifier_type, training_data, bigram = True)
        classifiers[classifier_type].append(bigram_classifier)
    return classifiers

# Compare the predictions with the right categories and returns
# if the prediction is correct or not, ordered by category
def calculate_results_by_category(predictions, target_categories):
    results = { 0:[], 1:[], 2:[] }
    counter = 0
    for target in target_categories:
        results[target].append(target == predictions[counter])
        counter += 1
    return results

# Compare the prediction with the real categories to calculate the performance
def calculate_performance_by_category(predictions, target_categories):
    results = calculate_results_by_category(predictions, target_categories)
    performance_by_category = { 0:[], 1:[], 2:[] }
    for category in results:
        performance_by_category[category] = np.mean(results[category])
    return performance_by_category

# Test the classifier with the test data and calculates the performance
def test_classifier(classifier, test_data):
    performance = {}
    predictions = classifier.predict(test_data.data)
    # Calculate the average performance of the classifier with the test data
    performance["mean"] = np.mean(predictions == test_data.target)
    # Calculate the performance of the classifier with the test data by category
    performance["by_category"] = calculate_performance_by_category(predictions, test_data.target)
    return performance

def show_results_head(classifier_type, bigram):
    if bigram:
        print("\nBigram " + classifier_type + " results")
    else:
        print("\nUnigram " + classifier_type+ " results")

def show_results(performance, category_names):
    mean = str(performance["mean"])
    print("Mean performance: " + mean)
    for category in performance["by_category"]:
        print(category_names[category])
        print(performance["by_category"][category])

# Load the datasets
training_data = load_dataset("train")
test_data = load_dataset("test")
category_names = test_data.target_names

# Build and train the classifiers
classifiers = build_and_train_all_classifiers(training_data)

# Test the classifiers and show the results
for classifier_type in classifiers:
    counter = 0
    for classifier in classifiers[classifier_type]:
        performance = test_classifier(classifier, test_data)
        show_results_head(classifier_type, counter==1)
        show_results(performance, category_names)
        counter += 1
