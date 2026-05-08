from src.models_trainer import trainLabelPropagationModel
from pathlib import Path
import csv
import statistics
import pandas as pd
from tqdm import tqdm
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
studyResult = pd.DataFrame(columns=['parameters','Avg precision', 'Stdev precision', 'Avg recall', 'Stdev recall', 'Avg accuracy', 'Stdev accuracy', 'Avg f1', 'Stdev f1'])
studyDetails = pd.DataFrame(columns=['parameters','Precisions', 'Recall', 'Accurancy', 'F1'])


project_root = Path.cwd()
folder = f'{project_root}/tests/hardcoded_secrets/parameters'


parameters = [['knn', 5, 20],
              ['knn', 6, 20],
              ['knn', 7, 20],
              ['knn', 8, 20],
              ['knn', 9, 20],
              ['rbf', 5, 18],
              ['rbf', 5, 19],
              ['rbf', 5, 20],
              ['rbf', 5, 21],
              ['rbf', 5, 22]
             ]


testingSet = f'{folder}/sets/test_set.csv'

for parameter in parameters:

    iteration = {'precision': [], 'recall': [], 'accuracy': [], 'f1': []}
    
    print(f"\n{'='*20} {parameter[0]}-{parameter[1]}-{parameter[2]} {'='*20}\n")


    for version in range(30):

        trainingSet = f'{folder}/sets/training_sets/training_set_{version}.csv'

        results = trainLabelPropagationModel(trainingSet, testingSet, None, kernel=parameter[0], n_neighbors=parameter[1], gamma=parameter[2])
        
        iteration['precision'].append(results['precision'])
        iteration['recall'].append(results['recall'])
        iteration['accuracy'].append(results['accuracy'])
        iteration['f1'].append(results['f1'])


    detailRow = [ str(parameter), str(iteration["precision"]), str(iteration["recall"]), str(iteration["accuracy"]), str(iteration["f1"])]
    row = [str(parameter), statistics.mean(iteration["precision"]), statistics.stdev(iteration["precision"]), statistics.mean(iteration["recall"]), statistics.stdev(iteration["recall"]), statistics.mean(iteration["accuracy"]), statistics.stdev(iteration["accuracy"]), statistics.mean(iteration["f1"]), statistics.stdev(iteration["f1"])]


    studyResult.loc[len(studyResult.index)] = row
    studyDetails.loc[len(studyDetails.index)] = detailRow


studyResult.to_csv(f"{folder}/results/results_matrix.csv")
studyDetails.to_csv(f"{folder}/results/results_matrix_details.csv")
print(studyResult)

