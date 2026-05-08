from src.models_trainer import trainLabelPropagationModel
from pathlib import Path
import pandas as pd
import statistics


studyResult = pd.DataFrame(columns=['parameters','Avg precision', 'Stdev precision', 'Avg recall', 'Stdev recall', 'Avg accuracy', 'Stdev accuracy', 'Avg f1', 'Stdev f1'])
studyDetails = pd.DataFrame(columns=['parameters','Precisions', 'Recall', 'Accurancy', 'F1'])


project_root = Path.cwd()
folder = f'{project_root}/tests/hardcoded_secrets/ratio'


splits = [
    'safe_90_vuln_10',
    'safe_80_vuln_20',
    'safe_70_vuln_30',
    'safe_60_vuln_40',
    'safe_50_vuln_50'
]

for split in splits:

    iteration = {'precision': [], 'recall': [], 'accuracy': [], 'f1': []}
    
    print(f"\n{'='*20} {split} {'='*20}\n")

    for version in range(30):

        trainingSet = f'{folder}/splits/training_sets/{split}/{split}_{version}.csv'
        testingSet = f'{folder}/splits/test_set.csv'

        results = trainLabelPropagationModel(trainingSet, testingSet, None, kernel="knn", n_neighbors=9)


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
