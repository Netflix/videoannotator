import functools
import itertools
import logging
import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, balanced_accuracy_score as ba
from tqdm.auto import tqdm

from . import config as cfg, data, models
from .data import LabeledDataset


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Metric:
    values: t.Tuple[float, ...]

    @property
    @functools.lru_cache()
    def values_finite(self) -> t.List[float]:
        if len(self.values) == 0:
            return []
        vals = [v for v in self.values if not np.isnan(v)]
        if len(vals) == 0:
            raise ValueError("Metric only contains NaN.")
        if len(vals) < len(self.values):
            logger.warning(f"Metric has NaN values: {self.values}")
        return vals

    @property
    def mean(self) -> float:
        return np.mean(self.values_finite).item() if len(self.values_finite) > 0 else 0

    @property
    def std(self) -> float:
        return np.std(self.values_finite).item() if len(self.values_finite) > 0 else 0

    def __repr__(self):
        return f"Metric(mean={self.mean:.2f}, std={self.std:.2f})"

    def add(self, x: float) -> "Metric":
        return Metric(values=self.values + (x,))

    @property
    def last(self) -> float:
        return self.values_finite[-1] if len(self.values_finite) > 0 else float("-inf")

    def ewma(self, alpha: float) -> float:
        assert 0 <= alpha <= 1
        if len(self.values_finite) == 0:
            return 0
        v = 1 - alpha
        n = len(self.values_finite)
        num = sum(x * (v**i) for i, x in enumerate(self.values_finite[::-1]))
        den = sum(v**i for i in range(n))
        return num / den


@dataclass(frozen=True)
class ExperimentResults:
    lds_train: LabeledDataset
    lds_validation: LabeledDataset
    average_precision: Metric
    balanced_accuracy: Metric


def run_experiment(
    model: models.Model,
    lds_train: LabeledDataset,
    lds_validation: LabeledDataset,  # TODO: take a list/dict instead?
    n_bootstrap: int = cfg.N_BOOTSTRAP,
) -> ExperimentResults:
    common_keys = set(lds_train.keys) & set(lds_validation.keys)
    if len(common_keys) > 0:
        raise ValueError(f"Train and validation sets have common keys: {common_keys}")
    model.fit(lds_train.x, lds_train.y)
    aps, bas = [], []
    for idx in range(n_bootstrap):
        x, y, _ = lds_validation.boostrap_xyk(idx=idx)
        bas.append(ba(y, model.predict(x)))
        aps.append(ap(y, model.predict_proba(x)))
    return ExperimentResults(
        lds_train=lds_train,
        lds_validation=lds_validation,
        average_precision=Metric(values=tuple(aps)),
        balanced_accuracy=Metric(values=tuple(bas)),
    )


def _ap_baseline(lds_val: LabeledDataset) -> Metric:
    thr = 1_000
    p = lds_val.pos_cnt
    n = len(lds_val)
    if n >= thr:
        return Metric(values=(p / n,))
    np.random.seed(0)
    y = [True] * p + [False] * (n - p)
    rand_scores = np.random.rand(thr, n)
    aps = tuple(ap(y, rand_scores[i]) for i in range(thr))
    return Metric(values=aps)


@dataclass(frozen=True)
class Experiment:
    ns: t.Tuple[int, ...] = cfg.AVE_EXPERIMENTS_NS
    methods: t.Tuple[str, ...] = cfg.AVE_EXPERIMENTS_METHODS
    seed: int = cfg.SEED
    scoring: t.Tuple[str, ...] = cfg.SCORING

    def experiment_active_learning(self, label: str) -> dict:
        # TODO: make sure zero-shot / random don't overlap with validation
        logger.info(f"Running active learning experiment for label={label}...")
        ds_agg = data.get_aggregate_labeled_dataset(label=label)
        ds_agg_train, ds_agg_val = ds_agg.split()
        # ds_val_lookup = dict(agg=ds_agg_val)
        res_final = dict()
        res_final["baseline_agg"] = _ap_baseline(lds_val=ds_agg_val)
        # TODO: remove disagreements and then run (if any positives left) => may still be misaligned
        # if label in cfg.LABELS_AVE:
        #     ds_ave = data.get_ave_validation_labeled_dataset(label=label)
        #     ds_val_lookup["ave"] = ds_ave
        #     logger.info("Zero shot with AVE data...")
        #     res_final["zero_shot_ave"] = run_experiment(
        #         model=models.ZeroShotText2Video(label=label),
        #         lds_train=ds_agg,
        #         lds_validation=ds_ave,
        #     )
        #     res_final["baseline_ave"] = _ap_baseline(lds_val=ds_ave)
        # else:
        #     logger.info(f"AVE data does NOT exist for label={label}")
        logger.info("Zero shot with agg data...")
        res_final["zero_shot"] = run_experiment(
            model=models.ZeroShotText2Video(label=label),
            lds_train=ds_agg_train,
            lds_validation=ds_agg_val,
        )
        val_keys = set(ds_agg_val.keys)
        ds_all = data.get_labeled_dataset_checkpoints(label=label)
        logger.info("Comparing...")
        res_final["cmp"] = self._run_experiments_comp(
            label=label,
            lds_validation=ds_agg_val,
        )
        res_final["checkpoints"] = [
            run_experiment(
                models.LogisticRegression(scoring="average_precision"),
                lds_train=ds.remove_keys(keys=val_keys),
                lds_validation=ds_agg_val,
            )
            for ds in tqdm(ds_all, desc="Processing checkpoints")
        ]
        return res_final

    def experiment_active_learning_batch(
        self,
        labels: t.Tuple[str, ...],
    ) -> dict:
        return {
            label: self.experiment_active_learning(label=label)
            for label in tqdm(labels)
        }

    def experiment_active_learning_aggregate(
        self, res_all: t.Dict[str, dict]
    ) -> pd.DataFrame:
        # TODO: AP vs. n => lines with CI: 1- baseline 2- zs 3- agg 4-
        pass

    @staticmethod
    def _get_lds_zero_shot(lds: LabeledDataset, n: int) -> LabeledDataset:
        zs = models.ZeroShotText2Video(label=lds.label)
        keys = zs.top_n_keys(lds=lds, n=n)
        return LabeledDataset(
            label=lds.label,
            pos=frozenset(k for k in keys if k in lds.pos),
            neg=frozenset(k for k in keys if k in lds.neg),
        )

    @staticmethod
    def _get_lds_train(
        label: str, n: int, method: str, keys_to_remove: t.FrozenSet[data.Key]
    ) -> LabeledDataset:
        ds = data.CompDataset(label=label, keys_to_remove=frozenset(keys_to_remove))
        if method == "random":
            return ds.random(n=n)
        elif method == "zero-shot-50-random-50":
            n_rand = round(n * 0.5)
            lds_rand = ds.random(n=n_rand)
            lds_zs = ds.zero_shot(n=n - n_rand)
            return lds_rand + lds_zs
        elif method == "zero-shot-20-random-80":
            n_zs = round(n * 0.2)
            lds_zs = ds.zero_shot(n=n_zs)
            lds_rand = ds.random(n=n - n_zs)
            return lds_rand + lds_zs
        elif method == "zero-shot":
            return ds.zero_shot(n=n)
        else:
            raise ValueError(f"method = {method} is not a valid choice.")

    @property
    def comp_iter(self) -> list:
        return list(
            itertools.product(
                self.ns, self.methods, ("average_precision", "balanced_accuracy")
            )
        )

    def _run_experiments_comp(
        self,
        label: str,
        lds_validation: LabeledDataset,
    ) -> dict:
        res = dict()
        for n, method, scoring in tqdm(self.comp_iter, desc="Running experiments"):
            try:
                lds_train = self._get_lds_train(
                    label=label,
                    n=n,
                    method=method,
                    keys_to_remove=frozenset(lds_validation.keys),
                )
                res[(n, method, scoring)] = run_experiment(
                    model=models.LogisticRegression(scoring=scoring),
                    lds_train=lds_train,
                    lds_validation=lds_validation,
                )
            except Exception as e:
                logger.error(
                    f"Experiment ({n}, {method}, {scoring}) for label={label} failed: {e}"
                )
        return res

    # def experiment_ave_self_comparison(
    #     self,
    #     label: str,
    # ) -> t.Dict[t.Tuple[int, str, str], ExperimentResults]:
    #     if label not in cfg.LABELS_AVE:
    #         raise ValueError(f"label={label} does not have AVE data")
    #     ds_ave = data.get_ave_validation_labeled_dataset(label=label)
    #     lds_train_base, lds_validation = ds_ave.split()
    #     return self._run_experiments_comp(
    #         label=label,
    #         lds_train_base=lds_train_base,
    #         lds_validation=lds_validation,
    #     )
    #
    # def experiment_ave_self_comparison_all(self) -> ...:
    #     return [
    #         self.experiment_ave_self_comparison(label=label) for label in cfg.LABELS_AVE
    #     ]
