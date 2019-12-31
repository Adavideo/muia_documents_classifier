# coding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
import numpy as np


# LOADING DATA FROM EXTERNAL FILES

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

def config_glossary():
    config_file = open("config.txt")
    config_data = config_file.read().split("\n")
    with_glossary = (config_data[3] == "True")
    print("With glossary: " + config_data[3])
    return with_glossary


# BUILDS AND TRAIN THE CLASSIFIERS

# Builds a CountVectorizer with the parameters especified
def build_vectorizer(bigram, with_glossary):
    if with_glossary:
        glossary = load_glossary()
        if bigram:
            vectorizer = CountVectorizer(vocabulary = glossary, ngram_range=(1, 2))
        else: # unigram
            vectorizer = CountVectorizer(vocabulary = glossary)
    else: # Without glossary
        if bigram:
            vectorizer = CountVectorizer(ngram_range=(1, 2))
        else: # unigram
            vectorizer = CountVectorizer()
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

def get_classifier_name(classifier_type, bigram, with_glossary):
    name = ""
    if not with_glossary:
        if bigram:
            name += "bigram "
        else:
            name += "unigram "
    name += classifier_type
    if with_glossary:
        name += " with glossary"
    else:
        name += " without glossary"
    return name

# Builds the type of classifier especified by the parameters
def build_classifier(classifier_type, with_glossary, bigram):
    classifier_name = get_classifier_name(classifier_type, bigram, with_glossary)
    print("Building classifier: " + classifier_name)
    vectorizer = build_vectorizer(bigram, with_glossary)
    if classifier_type == "Naive Bayes":
        classifier = naive_bayes_classifier(vectorizer)
    elif classifier_type == "SVM":
        classifier = svm_classifier(vectorizer)
    return classifier

def build_and_train_classifier(classifier_type, training_data, with_glossary, bigram):
    classifier = build_classifier(classifier_type, with_glossary, bigram)
    # Trains the classifier with the training data
    classifier.fit(training_data.data, training_data.target)
    return classifier

def build_and_train_all_classifiers(training_data, with_glossary):
    classifiers = {"Naive Bayes": [], "SVM": [] }
    if with_glossary:
        for classifier_type in classifiers:
            classifier = build_and_train_classifier(classifier_type, training_data, with_glossary, bigram = False)
            classifiers[classifier_type].append(classifier)
    else:
        for classifier_type in classifiers:
            # Building and training unigram classifier
            unigram_classifier = build_and_train_classifier(classifier_type, training_data, with_glossary, bigram = False)
            classifiers[classifier_type].append(unigram_classifier)
            # Building and training bigram classifier
            bigram_classifier = build_and_train_classifier(classifier_type, training_data, with_glossary, bigram = True)
            classifiers[classifier_type].append(bigram_classifier)
    return classifiers


# CALCULATE RESULTS

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


# SHOW RESULTS

def show_results_head(classifier_type, bigram, with_glossary):
    classifier_name = get_classifier_name(classifier_type, bigram, with_glossary)
    print("\nResults for " + classifier_name + " classifier")

def show_results(performance, category_names):
    mean = str(performance["mean"])
    print("Mean performance: " + mean)
    for category in performance["by_category"]:
        c = category_names[category]
        p = str(performance["by_category"][category])
        print(c + ": " + p)


# MAIN FUNCTION

# Load the datasets
training_data = load_dataset("train")
test_data = load_dataset("test")
category_names = test_data.target_names

# Reads from the config file if we use a glossary or not
with_glossary = config_glossary()

# Build and train the classifiers
print()
classifiers = build_and_train_all_classifiers(training_data, with_glossary)
print()

# Test the classifiers and show the results
for classifier_type in classifiers:
    counter = 0
    for classifier in classifiers[classifier_type]:
        performance = test_classifier(classifier, test_data)
        show_results_head(classifier_type, counter==1, with_glossary)
        show_results(performance, category_names)
        counter += 1

print()
