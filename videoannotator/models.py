import typing as t
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from . import config as cfg, data


@dataclass
class _ScikitBinaryClassifierAPI:
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        pass


@dataclass
class ZeroShotText2Video(_ScikitBinaryClassifierAPI):
    label: str

    @property
    def text_emb(self) -> np.ndarray:
        return np.array(data.get_text_embedding_dict()[self.label])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return cos(x, self.text_emb[np.newaxis, :]).squeeze()

    def predict(self, x: np.ndarray) -> np.ndarray:
        # doesn't really make sense, but we'll add for compatibility
        return np.array([False] * len(x))

    def top_n_keys(self, lds: data.LabeledDataset, n: int) -> t.List[data.Key]:
        scores = self.predict_proba(x=lds.x)
        return list(np.array(lds.keys)[scores.argsort()[::-1][:n]])


@dataclass
class LogisticRegression(_ScikitBinaryClassifierAPI):
    scoring: str
    n_splits: int = cfg.N_SPLITS
    n_jobs: int = cfg.N_JOBS
    seed: int = cfg.SEED
    max_iter: int = cfg.MAX_ITER

    def __post_init__(self):
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        self._model = LogisticRegressionCV(
            cv=cv,
            scoring=self.scoring,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(x)[:, 1]


Model = t.Union[
    LogisticRegression, RandomForestClassifier, XGBClassifier, ZeroShotText2Video
]
