"""
Usage:
    python -m src.models_trainer -l --lang puppet \
    -htrain data_extraction/per_language/puppet/hardcoded_secrets_dataset.csv \
    -htest tests/hardcoded_secrets/parameters/sets/test_set.csv
"""



from datasets import *
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import importlib
from sklearn.semi_supervised import LabelPropagation
import numpy as np
import csv
import pickle
from tqdm import tqdm
import argparse
import pandas as pd
import os

from src.vectorizer import getVector

def read_csv(file_path):
    data_array = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data_array.append(row)
    return data_array


def compute_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred,)
    f1 = f1_score(y_test, y_pred)
    return { 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def trainSetFitModel(datasets_path, trainedModel_path):
    dataset = load_dataset("csv", data_files=f"{datasets_path}/trainingSetComments.csv")
    test_dataset = load_dataset("csv", data_files=f"{datasets_path}/testSetComments.csv")

    ds = DatasetDict({
        'train': dataset['train'], 
        'test': test_dataset['train']})

    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2") 

    torch.cuda.empty_cache()

    trainer = SetFitTrainer(
        model=model,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        metric=compute_metrics,
        loss_class=CosineSimilarityLoss,
        num_iterations=20,
        num_epochs=5,
        column_mapping={"text": "text", "label": "label"},
    )
    trainer.train()
    
    model._save_pretrained(f'{trainedModel_path}/suspCommentsModel')

    metrics = trainer.evaluate()
    return metrics

def trainLabelPropagationModel(trainingDataset, testingDataset, trainedModel_path, kernel='knn', n_neighbors=7, gamma=20):
    
    training = pd.read_csv(trainingDataset, usecols=["word", "label"])
    training = training.dropna(subset=["word"])
    training["word"] = training["word"].astype(str)

    word_vocab = []
    word_label = []

    training = pd.read_csv(trainingDataset, usecols=["word", "label"])
    testing = pd.read_csv(testingDataset, usecols=["word", "label"])


    for _, row in tqdm(training.iterrows(), desc="Loading training set", unit="keyword"):
        word_vocab.append(row["word"])
        word_label.append(int(row["label"]))

    word_embeddings = []

    for word in tqdm(word_vocab, desc="Keyword Embedding", unit="keyword"):
        word_embeddings.append(getVector(word))

    word_embeddings = np.array(word_embeddings)

    modelSecrets = LabelPropagation(kernel=kernel, n_neighbors=n_neighbors, gamma=gamma)
    modelSecrets.fit(word_embeddings, word_label)

    if trainedModel_path:
        os.makedirs(trainedModel_path, exist_ok=True)
        with open(f'{trainedModel_path}/hardCodedSecretModel.pkl', 'wb') as f:
            pickle.dump(modelSecrets, f)

    predicted = []
    oracle_label = []

    for _, row in tqdm(testing.iterrows(), desc="Computing Metrics", unit="keyword"):
        oracle_label.append(int(row["label"]))
        predicted.append(modelSecrets.predict([getVector(row["word"])])[0])

    metrics = compute_metrics(oracle_label, predicted)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="This program retrain both the Label Propagation Model and the SetFit Model.")
    parser.add_argument( "-l", "--label", action="store_true", help="train only label propagation model.")
    parser.add_argument("-s", "--setfit", action="store_true", help="train only setfit model.")
    parser.add_argument("-b", "--both", action="store_true", help="train both models.")
    parser.add_argument("-htrain", "--hardcoded_training", type=str, help="Path to the training data.")
    parser.add_argument("-htest", "--hardcoded_testing",  type=str, help="Path to the testing data.")
    parser.add_argument("--lang", type=str, choices=["puppet", "chef"],
                        help="Language-specific model. Saves to pretrainedModels/{lang}/")

    args = parser.parse_args()

    if args.lang:
        trainedModel_path = f'./src/pretrainedModels/{args.lang}'
    else:
        trainedModel_path = './src/pretrainedModels'

    trainingDataset = args.hardcoded_training
    testingDataset = args.hardcoded_testing
    

    if not (args.label or args.setfit or args.both):
        parser.error('Choose which model train. Add -h for info')

    if args.label or args.both:
        
        print(f"Starting Label Propagation Training → {trainedModel_path}/")
        LabelPropagationMetrics = trainLabelPropagationModel(trainingDataset, testingDataset, trainedModel_path)
        print(f"LabelPropagation acuracy: {LabelPropagationMetrics['accuracy']}, precision: {LabelPropagationMetrics['precision']}, recall: {LabelPropagationMetrics['recall']}, f1: {LabelPropagationMetrics['f1']}\n\n" )
    
    if args.setfit or args.both:
        print(f"Starting SetFit Training → {trainedModel_path}/")
        SetFitMetrics = trainSetFitModel("datasets_path", trainedModel_path)
        print(f"SetFit acuracy: {SetFitMetrics['accuracy']}, precision: {SetFitMetrics['precision']}, recall: {SetFitMetrics['recall']}, f1: {SetFitMetrics['f1']}" )
    

if __name__ == "__main__":
    main()