import os
import re
import yaml

def replace_words(text, replacements):
    pattern = re.compile(r'{{ (.*?) }}')
    return pattern.sub(lambda x: replacements.get(x.group(1), x.group()), text)

def replace_words_in_file(input_file, output_file, replacements):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = replace_words(content, replacements)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

def replaceWords(files, replacements):
    for file_path in files:
       replace_words_in_file(file_path, file_path, replacements)

def find_ansible_file(folder):
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(('.yml', '.yaml')):
                file_list.append(os.path.join(root, file))
    return file_list

def findVars(files):
    pattern = r'\{\{([^{}]*)\}\}'
    matches = []

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            contenuto = file.read()

        matches.extend(re.findall(pattern, contenuto))

    final_list = list(set([elem.strip() for elem in matches]))
    return final_list

def findValues(files, varList):
    varValues = {elem :"" for elem in varList}
    for file_path in files:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            line = line.replace('-','').replace('"','').replace("'",'').rstrip().lstrip()
            if ':' in line:
                key, value = line.split(':',1)
                codePattern = r"[{.>|]"
                if key in varList:
                    if re.search(codePattern, value) is None: # this is because they both rapresent code execuitons
                        varValues[key]=(value.rstrip().lstrip())
    return varValues
                    

def main(folder_path):
    files = find_ansible_file(folder_path)
    results = findVars(files)

    patternCode = r"[ /[.|()]" # we are removing this becasue they rapresetn variables creatend
    singleWord = [elem for elem in results if re.search(patternCode, elem) is None ]

    values = findValues(files, singleWord)

    values = {key: values[key] for key in values if len(values[key]) > 0} # remove empty one
    replaceWords(files, values)