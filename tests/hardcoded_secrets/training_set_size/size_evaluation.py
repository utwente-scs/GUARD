from src.models_trainer import trainLabelPropagationModel
from pathlib import Path
import pandas as pd
import statistics


studyResult = pd.DataFrame(columns=['parameters','Avg precision', 'Stdev precision', 'Avg recall', 'Stdev recall', 'Avg accuracy', 'Stdev accuracy', 'Avg f1', 'Stdev f1'])
studyDetails = pd.DataFrame(columns=['parameters','Precisions', 'Recall', 'Accurancy', 'F1'])


project_root = Path.cwd()
folder = f'{project_root}/tests/hardcoded_secrets/training_set_size'


splits = [
    'size_100',
    'size_200',
    'size_500',
    'size_1000',
    'size_1500',
    'size_2000',
    'size_2500',
    'size_5000',
    'size_10000'
]

testingSet = f'{folder}/sizes/test_set.csv'


for split in splits:

    iteration = {'precision': [], 'recall': [], 'accuracy': [], 'f1': []}
    
    print(f"\n{'='*20} {split} {'='*20}\n")

    for version in range(30):

        trainingSet = f'{folder}/sizes/training_sets/{split}/{split}_{version}.csv'
        

        results = trainLabelPropagationModel(trainingSet, testingSet, None, kernel='knn', n_neighbors=9)


        iteration['precision'].append(results['precision'])
        iteration['recall'].append(results['recall'])
        iteration['accuracy'].append(results['accuracy'])
        iteration['f1'].append(results['f1'])

    detailRow = [ str(split), str(iteration["precision"]), str(iteration["recall"]), str(iteration["accuracy"]), str(iteration["f1"])]
    row = [str(split), statistics.mean(iteration["precision"]), statistics.stdev(iteration["precision"]), statistics.mean(iteration["recall"]), statistics.stdev(iteration["recall"]), statistics.mean(iteration["accuracy"]), statistics.stdev(iteration["accuracy"]), statistics.mean(iteration["f1"]), statistics.stdev(iteration["f1"])]


    studyResult.loc[len(studyResult.index)] = row
    studyDetails.loc[len(studyDetails.index)] = detailRow

studyResult.to_csv(f"{folder}/results/results_matrix.csv")
studyDetails.to_csv(f"{folder}/results/results_matrix_details.csv")
print(studyResult)
