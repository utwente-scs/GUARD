import re
import itertools
from gensim.models import KeyedVectors
from pathlib import Path


model_path = f'{Path.cwd()}/src/pretrainedModels/general/googleNews-vectors-negative300.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

def getVector(word: str):
    word = str(word).replace("0", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "").replace("6", "").replace("7", "").replace("8", "").replace("9", "")
    words = re.findall(r'[A-Z][a-z0-9]*|[a-z]+|\d+', word)
    composedVector = [0]*300
    
    for compWord in words:
        try:
            composedVector += word2vec_model[compWord.lower()]
        except:
            composedVector = [x + y for x, y in zip(composedVector, isComposed(compWord.lower()))] 

    return composedVector


def isComposed(word: str):
    subwords = []
    composedVector = [0]*300
    for subWord in generate_combinations(word):
            try:
                word2vec_model[subWord]
                subwords.append(subWord)
            except KeyError:
                pass
    longestWords = find_combinations(subwords, word)
    if longestWords != None:
        for combo in longestWords:
            composedVector += word2vec_model[combo]

    return composedVector


def find_combinations(subwords, original_word):
    subwords = [word for word in subwords if len(word) >= 2]
    for r in range(1, len(subwords) + 1):
        limited_combinations = itertools.islice(itertools.combinations(subwords, r), 10000)
        for combo in limited_combinations:
            if ''.join(combo) == original_word:
                return list(combo) 
    return None

def generate_combinations(string):
    n = len(string)
    combinations = []

    for i in range(0, n):
        for j in range(i+1, n+1):

            combinations.append(string[i:j])
    return combinations[:-1]
