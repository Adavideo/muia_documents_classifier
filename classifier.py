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

def build_vectorizer(bigram):
    if bigram:
        vectorizer = CountVectorizer(ngram_range=(1, 2))
    else:
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

# Train a classifier with the training data
def train_classifier(classifier, training_data):
    return classifier.fit(training_data.data, training_data.target)

def build_classifier(classifier_type, training_data, bigram):
    print("Building " + classifier_type + " classifier")
    vectorizer = build_vectorizer(bigram)
    if classifier_type == "Naive Bayes":
        classifier = naive_bayes_classifier(vectorizer)
    elif classifier_type == "SVM":
        classifier = svm_classifier(vectorizer)
    classifier = train_classifier(classifier, training_data)
    return classifier

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
classifiers = {"Naive Bayes": [], "SVM": [] }
nb_classifier_unigram = build_classifier("Naive Bayes",  training_data, bigram = False)
nb_classifier_bigram = build_classifier("Naive Bayes",  training_data, bigram = True)
classifiers["Naive Bayes"].append(nb_classifier_unigram)
classifiers["Naive Bayes"].append(nb_classifier_bigram)
svm_classifier_unigram = build_classifier("SVM",  training_data, bigram = False)
svm_classifier_bigram = build_classifier("SVM",  training_data, bigram = True)
classifiers["SVM"].append(svm_classifier_unigram)
classifiers["SVM"].append(svm_classifier_bigram)

# Test the classifiers
for classifier_type in classifiers:
    print(classifier_type)
    for classifier in classifiers[classifier_type]:
        performance = test_classifier(classifier, test_data)
        show_results(performance, category_names)
        print("\n")
