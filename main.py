import os
import io
import urllib.request
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns

from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, BayesianEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

# --- Configuration ---
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
LOCAL_FILENAME = "processed.cleveland.data"

# Column names for the Cleveland dataset (UCI)
COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# target: 0 = no disease, 1..4 = disease (we'll map to 0/1)
# --- Utilities ---


def download_dataset(url=DATA_URL, local=LOCAL_FILENAME, force=False):
    if os.path.exists(local) and not force:
        print(f"Using local copy: {local}")
        return local
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, local)
    print("Saved to", local)
    return local


def load_and_clean(path):
    # Load CSV (UCI file uses commas; missing values marked as '?')
    df = pd.read_csv(path, header=None, names=COLS, na_values='?')
    # Drop rows where target is missing (rare)
    df = df.dropna(subset=['target'])
    # Convert target to binary (0 no disease, 1 disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    # 'ca' and 'thal' sometimes have missing values -> impute
    # Impute numeric with median, categorical with most frequent
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer_num = SimpleImputer(strategy='median')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])
    return df


def preprocess_discretize(df, n_bins=5, strategy='quantile', encode='ordinal'):
    """
    Discretize continuous variables so Bayesian network learning & inference work on discrete nodes.
    Returns the discretized dataframe and discretizer object (for inverse transform if needed).
    """
    # Which columns to discretize (continuous clinical measurements)
    cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_cols = [c for c in df.columns if c not in cont_cols]

    # Fit discretizer
    disc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    df_cont = disc.fit_transform(df[cont_cols])
    df_cont = pd.DataFrame(df_cont, columns=cont_cols).astype(int)

    df_cat = df[cat_cols].copy()
    # Ensure categorical columns are ints (sex, cp, fbs, restecg, exang, slope, ca, thal, target)
    for c in df_cat.columns:
        df_cat[c] = df_cat[c].astype(int)

    df_disc = pd.concat([df_cont, df_cat], axis=1)
    # Reorder columns to keep same order as original
    df_disc = df_disc[df.columns]
    return df_disc, disc


def learn_structure(train_df, score_type='bic', max_iter=200):
    """
    Learn BN structure using HillClimbSearch.
    score_type: 'bic' or 'k2'
    """
    if score_type == 'k2':
        score = K2Score(train_df)
    else:
        score = BicScore(train_df)

    hc = HillClimbSearch(train_df, scoring_method=score)
    print("Searching for best model structure (this may take a while)...")
    best_model = hc.estimate(max_indegree=3, max_iter=max_iter)  # limit indegree for stable results
    print("Found edges:", best_model.edges())
    return best_model


def fit_parameters(model, train_df):
    """
    Fit CPDs (parameters) using BayesianEstimator (adds pseudo counts).
    """
    model = BayesianModel(model.edges())
    model.fit(train_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Use model to compute P(target=1 | evidence) for each test row and evaluate classification metrics.
    """
    infer = VariableElimination(model)
    probs = []
    for _, row in X_test.iterrows():
        evidence = row.to_dict()
        # Remove target from evidence
        evidence.pop('target', None)
        try:
            q = infer.query(variables=['target'], evidence=evidence, show_progress=False)
            p1 = q.values[1]  # P(target=1)
        except Exception as e:
            # If inference fails for some evidence combination, fallback to marginal
            q = infer.query(variables=['target'], show_progress=False)
            p1 = q.values[1]
        probs.append(p1)

    probs = np.array(probs)
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, ROC-AUC: {auc:.4f}")
    return probs, preds, {"accuracy": acc, "precision": prec, "recall": rec, "roc_auc": auc}


def plot_roc(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, probs):.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- Main flow ---
def main():
    # 1) download and load
    path = download_dataset()
    df = load_and_clean(path)
    print("Loaded data shape:", df.shape)
    print(df.head())

    # 2) discretize
    df_disc, discretizer = preprocess_discretize(df, n_bins=5, strategy='quantile')
    print("After discretization sample:\n", df_disc.head())

    # 3) train-test split
    train_df, test_df = train_test_split(df_disc, test_size=0.25, random_state=42, stratify=df_disc['target'])
    print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

    # 4) structure learning
    skeleton = learn_structure(train_df, score_type='bic', max_iter=200)

    # 5) parameter estimation
    model = fit_parameters(skeleton, train_df)
    print("Model CPDs learned.")

    # 6) evaluate
    probs, preds, metrics = evaluate_model(model, test_df.drop(columns=[]), test_df['target'])
    print("Metrics:", metrics)
    plot_roc(test_df['target'].values, probs)

    # 7) Example: predict for a new patient (raw values -> discretize -> predict)
    example_raw = {
        # raw continuous and categorical values in same format as original df (before discretize)
        "age": 54, "sex": 1, "cp": 2, "trestbps": 140, "chol": 239, "fbs": 0, "restecg": 1,
        "thalach": 151, "exang": 0, "oldpeak": 1.2, "slope": 2, "ca": 0, "thal": 2
    }

    # convert single patient to dataframe, preserve column order
    raw_df = pd.DataFrame([example_raw], columns=[c for c in df.columns if c != 'target'])
    # discretize continuous columns using fitted discretizer
    cont_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cont_disc = discretizer.transform(raw_df[cont_cols])
    cont_disc = pd.DataFrame(cont_disc, columns=cont_cols).astype(int)
    cat_df = raw_df.drop(columns=cont_cols).astype(int).reset_index(drop=True)
    example_disc = pd.concat([cont_disc, cat_df], axis=1)[df.columns.drop('target')]
    # inference
    infer = VariableElimination(model)
    q = infer.query(variables=['target'], evidence=example_disc.iloc[0].to_dict(), show_progress=False)
    print("Predicted P(heart_disease=1) for example patient:", q.values[1])

    # 8) Save model? (pgmpy allows pickling)
    import pickle
    with open("bayes_heart_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to bayes_heart_model.pkl")


if __name__ == "__main__":
    main()
