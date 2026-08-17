import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

DATASET_PATH = "../../resources/datasets/dataset_t3.csv"
GLITCH_IN = "../../resources/in/GLITCH_results.csv"
INTELLSIA_IN = "../../resources/in/intellisa_results.csv"

GLITCH_OUT = "./predictions/GLITCH_predictions_secrets_t3.csv"
INTELLSIA_OUT = "./predictions/intellsia_predictions_secrets_t3.csv"

def main():
    os.makedirs("./predictions", exist_ok=True)
    
    print(f"Loading dataset: {DATASET_PATH}")
    df_dataset = pd.read_csv(DATASET_PATH, keep_default_na=False, na_values=[])
    df_dataset["label"] = pd.to_numeric(df_dataset["label"], errors="coerce").fillna(-1).astype(int)
    
    df_labeled = df_dataset[df_dataset["label"] != -1].copy()
    unique_leaves = sorted(set(df_labeled["leaf"].astype(str)))
    print(f"Found {len(unique_leaves)} unique labeled leaves.")

    if os.path.exists(GLITCH_IN):
        df_glitch = pd.read_csv(GLITCH_IN, keep_default_na=False)
        glitch_positives = set(df_glitch["CODE"].astype(str))
        
        glitch_preds = [1 if leaf in glitch_positives else 0 for leaf in unique_leaves]
        
        pd.DataFrame({"leaf": unique_leaves, "pred": glitch_preds}).to_csv(GLITCH_OUT, index=False)
        print(f"GLITCH predictions successfully saved to: {GLITCH_OUT}")
    else:
        print(f"File not found: {GLITCH_IN}")

    if os.path.exists(INTELLSIA_IN):
        df_intell = pd.read_csv(INTELLSIA_IN, keep_default_na=False)
        intell_positives = set(zip(df_intell["PATH"].astype(str), df_intell["LINE"].astype(str)))
        
        path_col = next((col for col in ["path", "file", "filepath"] if col in df_dataset.columns), None)
        line_col = next((col for col in ["line", "line_number"] if col in df_dataset.columns), None)
        
        if path_col and line_col:
            df_labeled["_path_str"] = df_labeled[path_col].astype(str)
            df_labeled["_line_str"] = df_labeled[line_col].astype(str)
            
            flagged_leaves = set()
            for _, row in df_labeled.iterrows():
                if (row["_path_str"], row["_line_str"]) in intell_positives:
                    flagged_leaves.add(str(row["leaf"]))
                    
            intell_preds = [1 if leaf in flagged_leaves else 0 for leaf in unique_leaves]
        else:
            print("Dataset is missing path/line columns. IntellSia predictions defaulting to 0.")
            intell_preds = [0] * len(unique_leaves)
            
        pd.DataFrame({"leaf": unique_leaves, "pred": intell_preds}).to_csv(INTELLSIA_OUT, index=False)
        print(f"IntellSia predictions successfully saved to: {INTELLSIA_OUT}")
    else:
        print(f"File not found: {INTELLSIA_IN}")

if __name__ == "__main__":
    main()