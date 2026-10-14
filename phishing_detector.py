"""Training and inference helpers for the phishing email classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


PHISHING_LABEL = "Phishing Email"
SAFE_LABEL = "Safe Email"
DEFAULT_DATA_PATH = Path(__file__).parent / "datasets" / "Phishing_Email_Dataset.csv"


@dataclass(frozen=True)
class Prediction:
    label: str
    phishing_probability: float
    safe_probability: float


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Pipeline
    training_rows: int


def load_training_data(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the two columns required by the deployed model."""
    data = pd.read_csv(path, usecols=["Email Text", "Email Type"])
    data = data.dropna(subset=["Email Text", "Email Type"]).copy()
    data["Email Text"] = data["Email Text"].astype(str).str.strip()
    data = data[data["Email Text"] != ""]
    data = data.drop_duplicates(subset=["Email Text", "Email Type"])

    labels = set(data["Email Type"].unique())
    expected = {PHISHING_LABEL, SAFE_LABEL}
    if labels != expected:
        raise ValueError(f"Expected labels {sorted(expected)}, found {sorted(labels)}")
    return data


def build_pipeline(*, n_estimators: int = 200) -> Pipeline:
    """Build the tuned TF-IDF + Random Forest pipeline from the analysis."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=1_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=None,
                    bootstrap=False,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(path: Path | str = DEFAULT_DATA_PATH) -> ModelBundle:
    """Train on the repository dataset and return the fitted pipeline."""
    data = load_training_data(path)
    pipeline = build_pipeline()
    pipeline.fit(data["Email Text"], data["Email Type"])
    return ModelBundle(pipeline=pipeline, training_rows=len(data))


def predict_email(pipeline: Pipeline, email_text: str) -> Prediction:
    """Classify one email and return probabilities for both known classes."""
    cleaned_text = email_text.strip()
    if not cleaned_text:
        raise ValueError("Email text cannot be empty.")

    probabilities = pipeline.predict_proba([cleaned_text])[0]
    classifier = pipeline.named_steps["classifier"]
    scores = dict(zip(classifier.classes_, probabilities, strict=True))
    phishing_probability = float(scores[PHISHING_LABEL])
    safe_probability = float(scores[SAFE_LABEL])
    label = PHISHING_LABEL if phishing_probability >= 0.5 else SAFE_LABEL
    return Prediction(
        label=label,
        phishing_probability=phishing_probability,
        safe_probability=safe_probability,
    )
