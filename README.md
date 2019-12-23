# muia_documents_classifier
A documents classifier made for the class of Ingeniería Lingüística in the Master of Artificial Intelligence of Universidad Politécnica de Madrid (UPM)

To execute:
python3 classifier.py


To generate the documents for the dataset:

1- Copy the original files in /documents/original_documents and clean_documents.py in the same directory.

2- Execute for each of the categories (salud, politica and tecnologia):
	python clean_documents.py CATEGORY
		example: python clean_documents.py salud
   It will create the files salud_clean.txt, politica_clean.txt and tecnologia_clean.txt.
   They are a copy of the original files, but cleaned up. It has the Spanish characters fixed, some lines eliminated, and puts a marker ("===") to delimitate the diferent texts examples that we'll use to separate it in diferent documets in the next step.

Note: Delete the first line in the new documents. It only contains "===" and if you leave it there the first document created on the next step will be empty. 


3- Copy generate_documents.py in the directory.
   Create a folder for each category: salud, politica and tecnologia

4- Execute:
	python generate_documents.py
   It will generate documents for each text example in the folders of the categories.

5- Create the folders train_dataset and test_dataset in the same folder than classifier.py.
   Create subfolders for each category in both folders.
	Example: train_dataset/salud
   Copy there the examples that you want to use to train and to test. It is important that this examples are in subfolders of the corresponding category.
 
