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

def build_classifier(classifier_type, training_data):
    print("Building " + classifier_type + " classifier")
    if classifier_type == "Naive Bayes":
        classifier = naive_bayes_classifier()
    elif classifier_type == "SVM":
        classifier = svm_classifier()
    classifier = train_classifier(classifier,training_data)
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

def show_results(performance):
    mean = str(performance["mean"])
    print("Mean performance: " + mean)
    for category in performance["by_category"]:
        print("Category: " + str(category))
        print(performance["by_category"][category])


# Load the datasets
training_data = load_dataset("train")
test_data = load_dataset("test")

# Build and train the classifiers
nb_classifier = build_classifier("Naive Bayes",  training_data)
svm_classifier = build_classifier("SVM",  training_data)

# Test the classifiers
nb_performance = test_classifier(nb_classifier, test_data)
svm_performance = test_classifier(svm_classifier, test_data)

# Show the results
print ("\nNaive Bayes results")
show_results(nb_performance)
print ("\nSVM results")
show_results(svm_performance)
print("\n")
