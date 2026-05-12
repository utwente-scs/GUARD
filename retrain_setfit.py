"""
Retrain the SetFit suspicious comments model from susp_comments_dataset.csv.

Run from the GUARD/ root directory:
    python retrain_setfit.py

Prerequisites:
    pip install setfit==0.7.0 transformers==4.36.0 datasets==2.16.0 \
                sentence-transformers==2.2.2 huggingface_hub==0.20.3 torch scikit-learn pandas
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
    dataset_path = "datasets/susp_comments_dataset.csv"
    model_output_path = "src/pretrainedModels/suspCommentsModel"
    test_size = 0.2
    random_seed = 42
    n_per_class = 500  # SetFit is few-shot — 500 per class is plenty

    # --- Locate dataset ---
    if not os.path.exists(dataset_path):
        alt_paths = [
            "src/datasets/susp_comments_dataset.csv",
            "susp_comments_dataset.csv",
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                dataset_path = alt
                break
        else:
            print(f"ERROR: Could not find susp_comments_dataset.csv")
            print(f"Looked in: {[dataset_path] + alt_paths}")
            return

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # --- Rename column if needed ---
    if "word" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"word": "text"})
        print("Renamed column 'word' → 'text'")

    # --- Print full dataset distribution ---
    print(f"\nFull dataset size: {len(df)} examples")
    counts = df["label"].value_counts().sort_index()
    for label, count in counts.items():
        label_name = "suspicious" if label == 1 else "not suspicious"
        print(f"  label {label} ({label_name}): {count} ({count/len(df)*100:.1f}%)")

    # --- Subsample for SetFit (few-shot learner, doesn't need 150k examples) ---
    sampled = df.groupby("label").apply(
        lambda x: x.sample(n=min(n_per_class, len(x)), random_state=random_seed)
    ).reset_index(drop=True)
    print(f"\nSubsampled to {len(sampled)} examples ({n_per_class} per class)")

    # --- Stratified train/test split on the subsample ---
    train_df, test_df = train_test_split(
        sampled, test_size=test_size, random_state=random_seed, stratify=sampled["label"]
    )
    print(f"Train set: {len(train_df)} examples")
    print(f"Test set:  {len(test_df)} examples")

    # --- Hold out the rest as a generalization test set ---
    holdout_df = df.drop(sampled.index)
    print(f"Holdout set (generalization test): {len(holdout_df)} examples")

    # --- Save splits for reproducibility ---
    os.makedirs("datasets", exist_ok=True)
    train_df.to_csv("datasets/trainingSetComments.csv", index=False)
    test_df.to_csv("datasets/testSetComments.csv", index=False)
    holdout_df.to_csv("datasets/holdoutSetComments.csv", index=False)
    print("Saved splits to datasets/")

    # --- Convert to HuggingFace Datasets ---
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    ds = DatasetDict({"train": train_dataset, "test": test_dataset})

    # --- Train SetFit model ---
    print("\nLoading pretrained sentence transformer...")
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
    torch.cuda.empty_cache()

    print("Starting SetFit training (20 iterations, 5 epochs)...")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}\n")

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

    # --- Evaluate on test split ---
    metrics = trainer.evaluate()
    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS ({len(test_df)} examples)")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")

    # --- Evaluate on holdout set (generalization) ---
    print(f"\nRunning generalization evaluation on holdout set ({len(holdout_df)} examples)...")
    holdout_texts = holdout_df["text"].tolist()
    holdout_labels = holdout_df["label"].tolist()
    holdout_preds = model.predict(holdout_texts).tolist()
    holdout_metrics = compute_metrics(holdout_labels, holdout_preds)

    print(f"\n{'='*50}")
    print(f"HOLDOUT SET RESULTS ({len(holdout_df)} examples)")
    print(f"{'='*50}")
    print(f"  Accuracy:  {holdout_metrics['accuracy']:.4f}")
    print(f"  Precision: {holdout_metrics['precision']:.4f}")
    print(f"  Recall:    {holdout_metrics['recall']:.4f}")
    print(f"  F1-score:  {holdout_metrics['f1']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()