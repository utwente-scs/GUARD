import os
import re
import yaml
from collections import defaultdict

def replace_words(text, replacements):
    pattern = re.compile(r'\{\{\s*([^{}]*?)\s*\}\}')
    return pattern.sub(lambda m: replacements.get(m.group(1).strip(), m.group()), text)

def replace_words_in_file(input_file, output_file, replacements):
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    new_content = replace_words(content, replacements)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

def replaceWords(files, replacements):
    for file_path in files:
       replace_words_in_file(file_path, file_path, replacements)

def find_file(folder):
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(('.yml', '.yaml')):
                full_path = os.path.join(root, file)
                if os.path.isfile(full_path):
                    file_list.append(full_path)
    return file_list

def findVars(files):
    pattern = r'\{\{([^{}]*)\}\}'
    matches = []

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            contenuto = file.read()

        matches.extend(re.findall(pattern, contenuto))

    final_list = list(set([elem.strip() for elem in matches]))
    return final_list

def findValues(files, varList):
    varValues = defaultdict(set)

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            lines = file.readlines()
        for line in lines:
            line = line.replace('-','').replace('"','').replace("'",'').rstrip().lstrip()
            if ':' in line:
                key, value = line.split(':',1)
                codePattern = r"[{}|]|(?:^|\s)>\s*$"
                if key in varList:
                    if re.search(codePattern, value) is None:
                        varValues[key].add(value.strip())
    return varValues
                    

def main(folder_path):
    files = find_file(folder_path)
    results = findVars(files)

    patternCode = r"[ /[.|()]"
    singleWord = [elem for elem in results if re.search(patternCode, elem) is None ]

    values = findValues(files, singleWord)

    values = {key: values[key] for key in values if len(values[key]) > 0}
    replaceWords(files, values)