# coding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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
    elif parameter == "to classify":
        return config_data[4]
    elif parameter == "classified":
        return config_data[5]
    elif parameter == "separator":
        return config_data[6]

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

def build_classifier():
    classifier = Pipeline([('vectorizer', CountVectorizer()),
                            ('transformer', TfidfTransformer()),
                            ('svm_classifier', SGDClassifier(loss='hinge',
                                penalty='l2',alpha=1e-3, max_iter=20, random_state=42))])
    return classifier

def build_and_train_classifier(training_data):
    classifier = build_classifier()
    # Trains the classifier with the training data
    classifier.fit(training_data.data, training_data.target)
    return classifier

def write_doc(document, doc_name, path):
    out_filename = path + doc_name
    out_file = open(out_filename, 'w')
    print ("Writin in "+ out_filename)
    out_file.write(document)
    out_file.close()

def get_filename(complete_path, separator):
    parts = complete_path.split(separator)
    return parts[3]

def order_documents(predictions, to_classify, category_names):
    i = 0
    documents = to_classify.data
    path = config_path("classified")
    separator = config_path("separator")
    for predicted_category in predictions:
        category = category_names[predicted_category]
        category_path = path +  category + separator
        doc = documents[i]
        filename = get_filename(to_classify.filenames[i], separator)
        print("\nDocument " + filename + " classified in " + category)
        write_doc(doc, filename, category_path)
        i += 1

# Load the datasets
training_data = load_dataset("train")
to_classify = load_dataset("to classify")
category_names = training_data.target_names

# Build and train the classifiers
classifier = build_and_train_classifier(training_data)
predictions = classifier.predict(to_classify.data)

order_documents(predictions, to_classify, category_names)
