from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class RuntimeModel:
    def __init__(self, algo: str, degree: int = 3):
        self.algo = algo
        self.degree = degree
        self.pipeline = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("linreg", LinearRegression()),
            ]
        )

    @staticmethod
    def features(algo: str, ts_length, segment_length, candidate_step=None) -> np.ndarray:
        ts_length = np.atleast_1d(np.asarray(ts_length, dtype=float))
        segment_length = np.atleast_1d(np.asarray(segment_length, dtype=float))
        if algo == "pdss":
            step = segment_length if candidate_step is None else candidate_step
            step = np.broadcast_to(np.atleast_1d(np.asarray(step, dtype=float)), ts_length.shape)
            return np.column_stack([ts_length, segment_length, step])
        return np.column_stack([ts_length, segment_length])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RuntimeModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.pipeline.predict(X), a_min=0.0, a_max=None)

    def predict_one(self, ts_length: int, segment_length: int, candidate_step: int | None = None) -> float:
        X = self.features(self.algo, ts_length, segment_length, candidate_step)
        return float(self.predict(X)[0])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"algo": self.algo, "degree": self.degree, "pipeline": self.pipeline}, path)

    @classmethod
    def load(cls, path: Path) -> "RuntimeModel":
        data = joblib.load(Path(path))
        model = cls(data["algo"], data["degree"])
        model.pipeline = data["pipeline"]
        return model
