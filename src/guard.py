import csv
import re
from setfit import SetFitModel
import importlib
import csv
import pickle
import argparse
import os
import shutil
import contexualizationPhase

import time

def find_ansible_file(folder):
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(('.yml', '.yaml')):
                file_list.append(os.path.join(root, file))
    return file_list

def read_csv(file_path):
    data_array = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data_array.append(row)
    return data_array

def fileReader(filePath):
    with open(filePath, 'r') as file:
        lines = file.readlines()
    return lines

def writer_csv(dataset, file):
    with open(file, mode='w', newline='') as file_csv:
        writer = csv.writer(file_csv)
        for line in dataset:
            writer.writerow(line)

def guard_file(filePath, smells):
    lines = fileReader(filePath)
    filename = filePath.replace("./temp/", "")
    smell_detected = False

    for line_number, row in enumerate(lines, start=1):
        row = row.rstrip()
        if '---' in row or not row.strip():
            continue

        if '#' in row:
            divided = row.split('#', 1)
            comment = divided[1].lstrip() if len(divided) > 1 else ""
            line = divided[0]
            commentPredict = modelComments.predict(comment)
            if commentPredict > 0:
                smells.append([filename, 'suspicious comment', line_number, row])
                smell_detected = True
        else:
            line = row.lstrip('-').lstrip()

        if ':' in line:
            key, value = line.split(':', 1)
            value = value.lstrip()
            
            if re.search(errors['invalindBinding'], value):
                smells.append([filename, 'improper ip address binding', line_number, line])
                smell_detected = True
            elif 'http' in value:
                if re.search(errors['http'], value) and not any(item in value for item in httpWhiteList):
                    smells.append([filename, 'use of http', line_number, line])
                    smell_detected = True
            elif any(re.search(weak, value) for weak in errors['weakCrypto']):
                smells.append([filename, 'weak crypto', line_number, line])
                smell_detected = True
            else:
                secretPredict = modelSecrets.predict([vectorizer.getVector(key)])
                if secretPredict > 0 and ('{{' not in value) and len(value) > 0:
                    smells.append([filename, 'hardcoded-secret', line_number, line])
                    smell_detected = True

    if not smell_detected:
        smells.append([filename, 'none', 'none'])

def guard_files(output):
    smells = []

    global vectorizer
    global modelComments
    global modelSecrets

    loadtime = time.time()
    vectorizer = importlib.import_module("vectorizer")
    modelComments =  SetFitModel._from_pretrained('./pretrainedModels/suspCommentsModel')
    with open('./pretrainedModels/hardCodedSecretModel.pkl', 'rb') as f:
        modelSecrets = pickle.load(f)
    loadModelTime = time.time() - loadtime

    
    for filePath in find_ansible_file('./temp'):
        guard_file(filePath, smells)
    

    writer_csv(smells, output)


def main():
    parser = argparse.ArgumentParser(description="GUARD: Robust Detection of Ansible Security Smells.")
    parser.add_argument( "-p", "--path", help="path is the file or folder to analyze.", required=True)
    parser.add_argument("-f", "--folder", action="store_true", help="specify path is a folder.")
    parser.add_argument("-c", "--context", action="store_true", help="activate contextualizzation procedure.")
    parser.add_argument("-o", "--output", help="name of output file.", default='output.csv')

    args = parser.parse_args()

    global errors
    global httpWhiteList

    errors = {
        'invalindBinding': r'0.0.0.0',
        'weakCrypto': [r'md5', r'sha1', r'arcfour'],
        'http': r'http://'
    }

    httpWhiteList = ['localhost', '127.0.0.1']
    

    if args.context and not args.folder:
        parser.error("Contextualization phase only work with projects not single files.")


    workingDirectory = './temp'
    
    try:
        if args.folder:
            shutil.copytree(args.path, workingDirectory)
            if args.context:
                contexualizationPhase.main('./temp')
        
        else:
            os.mkdir(workingDirectory)
            shutil.copy(args.path, workingDirectory)
    
        start_time = time.time()
        guard_files(args.output)
        final_time = time.time() - start_time

        shutil.rmtree('./temp')

    
    except IsADirectoryError as d:
        print(d)
        shutil.rmtree('./temp')


if __name__ == '__main__':
    main()