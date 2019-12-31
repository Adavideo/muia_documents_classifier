# Documents Classifier
A documents classifier made for the class of Ingeniería Lingüística in the Master of Artificial Intelligence of Universidad Politécnica de Madrid (UPM)

There is two classifiers. 

Classifier 1: 

Trains differents classifiers and performs tests with the testing dataset to see the performance.

To execute:

	python3 classifier1.py


Classifier 2:

Takes the files to classify from the folder especified in the config file (more information below), and orders them on another folder (also especified in the config file) into the predicted category. 
Note: It does not erase the files from the original folder, just makes copies of them. 

To execute:

	python3 classifier2.py


Config file:

Some parameters can be configured editing the file config.txt

	1st line: path for the training dataset.
	2nd line: path for the test dataset.
	3rd line: path of the glossary file.
	4th line: True to use the glossary, False to don't use it.
	5th line: path for the files to classify in the second classifier
	6th line: path for the folder to put the classified documents
	7th line: character that separate folders on paths. "/" or "\" depending on the operating system.

The paths are by default writen for linux. They should be edited before executing the program in a diferent operating system.


---

To generate the documents for the dataset:


1- Copy in the same directory the files in:

	/documents/original_documents

And the scripst:

	clean_documents.py
	generate_documents.py


2- Execute for each of the categories (salud, politica and tecnologia):

	python clean_documents.py CATEGORY

example:

	python clean_documents.py salud


It will create the files salud_clean.txt, politica_clean.txt and tecnologia_clean.txt.

They are a copy of the original files, but cleaned up. It has the Spanish characters fixed, some lines eliminated, and puts a marker ("===") to delimitate the diferent texts examples that we'll use to separate it in diferent documets in the next step.

	Note: Delete the first line in the new documents that contains "===".
	It causes the first document created on the next step to be empty.
	More info: https://github.com/Adavideo/muia_documents_classifier/issues/2


3- Create a folder for each category: salud, politica and tecnologia


4- Execute:

	python generate_documents.py

It will generate documents for each text example in the folders of the categories.


5- Create the folders train_dataset and test_dataset in the same folder than classifier.py.

Create subfolders for each category in both folders.

	Example: train_dataset/salud


6- Copy there the examples that you want to use to train and to test.

It is important that this examples are in subfolders of the corresponding category.
