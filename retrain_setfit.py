"""
Retrain the SetFit suspicious comments model from susp_comments_dataset.csv.

Run from the GUARD/ root directory:
    python retrain_setfit.py

Prerequisites:
    pip install setfit==0.7.0 transformers==4.36.0 datasets==2.16.0 \
                sentence-transformers==2.2.2 huggingface_hub==0.20.3 torch

What this script does:
    1. Reads susp_comments_dataset.csv (columns: word, label)
    2. Prints class distribution so you can sanity-check
    3. Renames 'word' → 'text' (SetFit expects 'text')
    4. Stratified 80/20 train/test split
    5. Trains SetFit model (paraphrase-mpnet-base-v2, 20 iterations, 5 epochs)
    6. Saves trained model to pretrainedModels/suspCommentsModel/
    7. Prints evaluation metrics (accuracy, precision, recall, F1)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset, DatasetDict
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
import torch


def compute_metrics(y_test, y_pred):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def main():
    # --- Configuration ---
    dataset_path = "datasets/susp_comments_dataset.csv"  # adjust if different location
    model_output_path = "pretrainedModels/suspCommentsModel"
    test_size = 0.2
    random_seed = 42

    # --- Load dataset ---
    if not os.path.exists(dataset_path):
        # Try alternative locations
        alt_paths = [
            "src/datasets/susp_comments_dataset.csv",
            "datasets/susp_comments_dataset.csv",
            "susp_comments_dataset.csv",
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                dataset_path = alt
                break
        else:
            print(f"ERROR: Could not find susp_comments_dataset.csv")
            print(f"Looked in: {[dataset_path] + alt_paths}")
            print("Pass the correct path by editing dataset_path in this script.")
            return

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # --- Rename column if needed ---
    if "word" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"word": "text"})
        print("Renamed column 'word' → 'text'")

    # --- Print class distribution ---
    print(f"\nDataset size: {len(df)} examples")
    print(f"Class distribution:")
    counts = df["label"].value_counts().sort_index()
    for label, count in counts.items():
        label_name = "suspicious" if label == 1 else "not suspicious"
        print(f"  label {label} ({label_name}): {count} ({count/len(df)*100:.1f}%)")

    # --- Stratified train/test split ---
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=df["label"]
    )
    print(f"\nTrain set: {len(train_df)} examples")
    print(f"Test set:  {len(test_df)} examples")

    # --- Also save the splits as CSVs (for reproducibility & future use) ---
    os.makedirs("datasets", exist_ok=True)
    train_df.to_csv("datasets/trainingSetComments.csv", index=False)
    test_df.to_csv("datasets/testSetComments.csv", index=False)
    print("Saved train/test splits to datasets/trainingSetComments.csv and datasets/testSetComments.csv")

    # --- Convert to HuggingFace Datasets ---
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    ds = DatasetDict({"train": train_dataset, "test": test_dataset})

    # --- Train SetFit model ---
    print("\nLoading pretrained sentence transformer...")
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

    torch.cuda.empty_cache()

    print("Starting SetFit training (20 iterations, 5 epochs)...")
    print("This may take a while depending on dataset size and hardware.\n")

    trainer = SetFitTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        metric=compute_metrics,
        loss_class=CosineSimilarityLoss,
        num_iterations=20,
        num_epochs=5,
        column_mapping={"text": "text", "label": "label"},
    )
    trainer.train()

    # --- Save model ---
    os.makedirs(model_output_path, exist_ok=True)
    model._save_pretrained(model_output_path)
    print(f"\nModel saved to: {model_output_path}/")

    # --- Evaluate ---
    metrics = trainer.evaluate()
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()