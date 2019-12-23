# coding: utf-8
from sys import argv

script, filename = argv
in_filename = filename + ".txt"
out_filename = filename + "_clean.txt"
in_file = open(in_filename)
out_file = open(out_filename, 'w')
text_lines = in_file.readlines()

def ignore(line):
    lines_to_ignore = ["Ver noticia en formato original... ",
                        "Términos: a ", "Resumen de Prensa",
                        "(Portaltic/EP)",
                        "? [V",
                        "12/12/2019 12:18", "22/12/2019 11:02", "12/12/2019 12:11"
                        ]
    for ignore in lines_to_ignore:
        if ignore in line:
            return True
    if "página " in line and ( "/112" in line or "/107" in line or "/100" in line):
        return True
    return False

def is_news_head(line):
    head = "Medio: "
    if head in line:
        return True
    else:
        return False

def fix_characters(line):
    special_characters = [["Ã±","ñ"],["Ã¡","á"],["Ã©","é"],["Ã3","ó"], ["Ão","ú"], ["Ã","í"], ["Â",""]]
    for character in special_characters:
        if character[0] in line:
            line = line.replace(character[0],character[1])
    return line

for line in text_lines:
    if not ignore(line):
        if is_news_head(line):
            out_file.write("===\n")
            print("\n")
        else:
            fixed_line = fix_characters(line)
            out_file.write(fixed_line)
            print(fixed_line)


out_file.close()
in_file.close()
