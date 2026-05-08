"""
Spam Email/SMS Classifier - Training Script

This script:
- Loads the Kaggle/UCI SMS Spam Collection dataset from spam.csv
- Cleans the data (drops duplicates)
- Vectorizes text using TF-IDF
- Trains multiple classical ML models
- Evaluates them with Accuracy, Classification Report, and ROC AUC
- Selects the best model and saves it + the vectorizer to the model/ folder

Constraints (per project requirements):
- Only use: scikit-learn, pandas, numpy, pickle (built-in)
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class ModelResult:
    """Holds evaluation artifacts for one trained model."""

    name: str
    model: Any
    accuracy: float
    roc_auc: float
    fpr: np.ndarray
    tpr: np.ndarray


def load_dataset(path: str = "spam.csv") -> pd.DataFrame:
    """
    Load the dataset exactly as required.

    Why latin-1:
    - The original dataset contains some characters that can break strict UTF-8 decoding.

    Why usecols:
    - The CSV contains extra unnamed columns; we only want label/message.
    """

    df = pd.read_csv(path, encoding="latin-1", usecols=["v1", "v2"])
    df = df.rename(columns={"v1": "label", "v2": "message"})
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset.

    - Drop duplicates:
      Duplicate messages can bias evaluation and inflate metrics, so we remove them.
    - Drop missing:
      Ensure every row has a message and label.
    """

    df = df.dropna(subset=["label", "message"])
    df = df.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string labels to integers.

    Required mapping:
    - spam -> 1
    - ham  -> 0
    """

    label_map = {"ham": 0, "spam": 1}
    df = df.copy()
    df["label"] = df["label"].map(label_map)
    if df["label"].isna().any():
        bad = df[df["label"].isna()]
        raise ValueError(
            "Found unexpected labels. Expected only 'ham' and 'spam'. "
            f"Examples:\n{bad.head(5)}"
        )
    df["label"] = df["label"].astype(int)
    return df


def build_vectorizer() -> TfidfVectorizer:
    """
    Create the TF-IDF vectorizer exactly as required.

    - stop_words='english' removes common English words that usually add noise.
    - max_features=5000 limits vocabulary size to keep the model lightweight and fast.
    """

    return TfidfVectorizer(stop_words="english", max_features=5000)


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Create the 5 required models.

    Note on "XGBoost":
    - The real XGBoost library is NOT allowed by the dependency constraints.
    - We use a strong boosting-style baseline available in scikit-learn: HistGradientBoostingClassifier.
      This fills the same niche (tree boosting) while respecting the library restrictions.
    """

    # Import here to keep the dependency surface explicit and avoid unused imports
    from sklearn.ensemble import HistGradientBoostingClassifier

    return {
        "MultinomialNB": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "XGBoost (sklearn HistGradientBoosting)": HistGradientBoostingClassifier(
            random_state=random_state
        ),
    }


def evaluate_model(
    name: str,
    model: Any,
    X_test,
    y_test: np.ndarray,
) -> ModelResult:
    """
    Evaluate one fitted model.

    Metrics required:
    - Accuracy
    - Classification report (precision/recall/f1)
    - ROC AUC and ROC curve points (FPR/TPR)
    """

    # Predicted class labels for accuracy + classification report
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    # Predicted probabilities for ROC AUC + ROC curve
    # Most models here support predict_proba; if one doesn't, we fall back to decision_function.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        # Convert unbounded scores to (0,1) for AUC stability
        y_score = 1.0 / (1.0 + np.exp(-raw))
    else:
        raise TypeError(f"Model {name} does not support probability/scoring needed for ROC AUC.")

    roc_auc = float(roc_auc_score(y_test, y_score))
    fpr, tpr, _thresholds = roc_curve(y_test, y_score)

    print("\n" + "=" * 80)
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham (0)", "spam (1)"]))
    print("ROC Curve Points (first 10):")
    for i in range(min(10, len(fpr))):
        print(f"  point {i+1:02d}: FPR={fpr[i]:.4f}, TPR={tpr[i]:.4f}")

    return ModelResult(
        name=name, model=model, accuracy=acc, roc_auc=roc_auc, fpr=fpr, tpr=tpr
    )


def pick_best(results: Dict[str, ModelResult]) -> ModelResult:
    """
    Pick the best model.

    Why ROC AUC:
    - Accuracy can look deceptively good if the dataset is imbalanced.
    - ROC AUC evaluates ranking quality across thresholds, which is useful for spam filtering.
    """

    return max(results.values(), key=lambda r: (r.roc_auc, r.accuracy))


def ensure_model_dir(path: str = "model") -> None:
    """Ensure the output directory exists before writing artifacts."""

    os.makedirs(path, exist_ok=True)


def save_artifacts(best: ModelResult, vectorizer: TfidfVectorizer, out_dir: str = "model") -> None:
    """
    Save the chosen model + vectorizer using pickle.

    Why save both:
    - The model expects TF-IDF features with the exact same vocabulary learned at training time.
    """

    ensure_model_dir(out_dir)

    model_path = os.path.join(out_dir, "model.pkl")
    vect_path = os.path.join(out_dir, "vectorizer.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best.model, f)

    with open(vect_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n" + "-" * 80)
    print(f"Saved best model to: {model_path}")
    print(f"Saved vectorizer to: {vect_path}")
    print(f"Best model: {best.name} (ROC AUC={best.roc_auc:.4f}, Accuracy={best.accuracy:.4f})")


def main() -> None:
    # 1) Load dataset
    df = load_dataset("spam.csv")

    # 2) Clean dataset
    df = clean_dataset(df)

    # 3) Encode labels
    df = encode_labels(df)

    # 4) Train/test split (80/20) with fixed seed for reproducibility
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["message"].values,
        df["label"].values,
        test_size=0.2,
        random_state=42,
        stratify=df["label"].values,  # keep spam/ham ratio similar across splits
    )

    # 5) Vectorize text (fit on train only; transform on train + test)
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # 6) Train and evaluate 5 models
    models = get_models(random_state=42)
    results: Dict[str, ModelResult] = {}

    for name, model in models.items():
        # Fit on training data
        #
        # Important:
        # - TF-IDF produces a sparse matrix.
        # - Some estimators (notably histogram-based gradient boosting) require dense arrays.
        # - To keep memory reasonable, we only densify when needed.
        X_train_fit = X_train
        X_test_eval = X_test
        try:
            model.fit(X_train_fit, y_train)
        except TypeError as e:
            msg = str(e).lower()
            if "sparse matrix" in msg and "dense" in msg:
                X_train_fit = X_train.toarray().astype(np.float32, copy=False)
                X_test_eval = X_test.toarray().astype(np.float32, copy=False)
                model.fit(X_train_fit, y_train)
            else:
                raise

        # Evaluate on test data
        result = evaluate_model(name=name, model=model, X_test=X_test_eval, y_test=y_test)
        results[name] = result

    # 7) Select the best model and save artifacts
    best = pick_best(results)
    save_artifacts(best=best, vectorizer=vectorizer, out_dir="model")


if __name__ == "__main__":
    main()

