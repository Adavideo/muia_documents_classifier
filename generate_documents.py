# coding: utf-8
from sys import argv

script, category = argv
path = category + "/"

in_filename = category + "_clean.txt"

in_file = open(in_filename)


text = in_file.read()
documents = text.split("===")

count = 1
for doc in documents:
    out_filename = path + category + "_" + str(count) + ".txt"
    out_file = open(out_filename, 'w')
    print ("\nWritin in "+ out_filename)
    print (doc)
    out_file.write(doc)
    out_file.close()
    count += 1

in_file.close()
