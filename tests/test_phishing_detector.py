import tempfile
import unittest
from pathlib import Path

import pandas as pd

from phishing_detector import (
    PHISHING_LABEL,
    SAFE_LABEL,
    build_pipeline,
    load_training_data,
    predict_email,
)


class PhishingDetectorTests(unittest.TestCase):
    def setUp(self):
        self.training_data = pd.DataFrame(
            {
                "Email Text": [
                    "urgent verify password account immediately",
                    "click link confirm banking credentials",
                    "invoice attachment payment overdue urgent",
                    "team meeting agenda for tomorrow",
                    "thanks for the project update",
                    "family dinner plans this weekend",
                ],
                "Email Type": [
                    PHISHING_LABEL,
                    PHISHING_LABEL,
                    PHISHING_LABEL,
                    SAFE_LABEL,
                    SAFE_LABEL,
                    SAFE_LABEL,
                ],
            }
        )
        self.pipeline = build_pipeline(n_estimators=10)
        self.pipeline.fit(self.training_data["Email Text"], self.training_data["Email Type"])

    def test_prediction_probabilities_sum_to_one(self):
        result = predict_email(self.pipeline, "urgent password verification link")
        self.assertIn(result.label, {PHISHING_LABEL, SAFE_LABEL})
        self.assertAlmostEqual(
            result.phishing_probability + result.safe_probability,
            1.0,
        )

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            predict_email(self.pipeline, "   ")

    def test_training_data_validation_and_deduplication(self):
        duplicated = pd.concat([self.training_data, self.training_data.iloc[[0]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            duplicated.to_csv(path, index=False)
            loaded = load_training_data(path)
        self.assertEqual(len(loaded), len(self.training_data))


if __name__ == "__main__":
    unittest.main()
