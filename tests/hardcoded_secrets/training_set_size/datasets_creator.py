import os
import pandas as pd
import random
from pathlib import Path

# Function to create a test set
def create_test_set(secret_rows, safe_rows, secret_sample_size=50, safe_sample_size=450, seed=None):
    # Set the seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    
    # Ensure there are enough rows in the source datasets
    if len(secret_rows) < secret_sample_size or len(safe_rows) < safe_sample_size:
        raise ValueError("Not enough rows to create the test set.")
    
    # Sample rows for the test set
    secret_test_set = secret_rows.sample(secret_sample_size, random_state=random.randint(0, 10000))
    safe_test_set = safe_rows.sample(safe_sample_size, random_state=random.randint(0, 10000))
    
    # Concatenate secret and safe samples to create the test set
    test_set = pd.concat([secret_test_set, safe_test_set])
    
    # Return the test set and the remaining rows
    return test_set, secret_rows.drop(secret_test_set.index), safe_rows.drop(safe_test_set.index)

# Function to create training sets
def create_training_sets(secret_rows, safe_rows, perc_safe,  sizes,  num_versions, project_root, seed=None ):
    if seed is not None:
        random.seed(seed)
    
    for size in sizes:
        for version in range(num_versions):
            secret_sample_size = int(size * (100 - perc_safe) / 100)
            safe_sample_size = int(size * perc_safe / 100)
            
            # Creation of the training set
            training_set = pd.concat([
                secret_rows.sample(secret_sample_size, random_state=random.randint(0, 10000)),
                safe_rows.sample(safe_sample_size, random_state=random.randint(0, 10000))
            ])

            # Build output folder path 
            output_folder = os.path.join(project_root, "tests", "hardcoded_secrets", "training_set_size", "sizes", "training_sets" , f"size_{size}")
            os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
            
            # Full path to output CSV
            output_file = os.path.join(output_folder, f"size_{size}_{version}.csv")
            
            # Save CSV
            training_set.to_csv(output_file, index=False, header=True)
            


# Main script logic
if __name__ == "__main__":

    project_root = Path.cwd()
    dataset = f"{project_root}/hardcoded_secrets_dataset.csv"
    
    dataset_rows = pd.read_csv(dataset, usecols=["word", "label"])

    secret_rows = dataset_rows[dataset_rows['label'] == 1]
    safe_rows = dataset_rows[dataset_rows['label'] == 0]

    random_seed = 42
    test_set, secret_rows_remaining, safe_rows_remaining = create_test_set(
        secret_rows, safe_rows, seed=random_seed
    )
    test_set.to_csv(f'{project_root}/tests/hardcoded_secrets/training_set_size/sizes/test_set.csv', index=False, header=True)
    
    sizes = [100, 200, 500, 1000, 1500, 2000, 2500, 5000, 10000]
    
    perc_safe = 70
    versions = 30

    create_training_sets(secret_rows_remaining, safe_rows_remaining, perc_safe, sizes, versions, project_root, seed=random_seed)
