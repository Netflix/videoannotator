import collections
import fractions
import functools
import json
import mmh3
import logging
import typing as t
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd

from . import config as cfg, io

logger = logging.getLogger(__name__)

Key = str
Embedding = np.ndarray
Label = str


@functools.lru_cache(maxsize=1)
def get_embeddings_dict() -> t.Dict[Key, Embedding]:
    logger.info("Loading embeddings for the first time...")
    keys = pd.read_csv(io.PATHS_STATIC.shot_data, usecols=["key"]).key.tolist()
    with h5py.File(io.PATHS_STATIC.embeddings, "r") as hf:
        embs = hf["embeddings"][:]
    d = {key: emb for key, emb in zip(keys, embs)}
    logger.info("Finished loading embeddings.")
    return d


@functools.lru_cache(maxsize=1)
def get_text_embedding_dict() -> t.Dict[Key, Embedding]:
    logger.info("Loading text embeddings for the first time...")
    data = json.load(open(io.PATHS_STATIC.text_embeddings))
    logger.info("Finished loading text embeddings.")
    return data


@functools.lru_cache(maxsize=1)
def get_key_to_cluster_idx_dict() -> t.Dict[Key, int]:
    df = pd.read_csv(io.PATHS_STATIC.shot_data)
    return df.set_index("key").cluster_index.to_dict()


@dataclass(frozen=True)
class LabeledDataset:
    label: str
    pos: t.FrozenSet[Key]
    neg: t.FrozenSet[Key]
    _cluster_cnt: int = cfg.CLUSTER_CNT

    def __post_init__(self):
        if len(self) == 0:
            raise ValueError(f"Dataset with label={self.label} has no data.")
        common = self.pos & self.neg
        if len(common) > 0:
            raise ValueError(
                f"keys cannot be in both positive and negative sets. these keys are in both: {common}"
            )
        elif self.pos_cnt == 0:
            logger.warning(f"{self} has no positive annotations.")
        elif self.neg_cnt == 0:
            logger.warning(f"{self} has no negative annotations.")

    def __len__(self) -> int:
        return len(self.pos) + len(self.neg)

    def __repr__(self) -> str:
        return (
            f"LabeledDataset(label={self.label}, cnt={len(self)}, pos_cnt={self.pos_cnt}, "
            f"neg_cnt={self.neg_cnt}, pos_rate={self.pos_rate:.2f})"
        )

    @property
    def keys(self) -> t.List[Key]:
        return self.keys_pos + self.keys_neg

    @property
    def y(self) -> np.array:
        return np.array([True] * self.pos_cnt + [False] * self.neg_cnt)

    @property
    @functools.lru_cache()
    def x(self) -> np.ndarray:
        return np.vstack([get_embeddings_dict()[key] for key in self.keys])

    @property
    @functools.lru_cache()
    def data_score(self) -> float:
        k2c = get_key_to_cluster_idx_dict()
        part_pos = self._data_score_part(
            keys=self.keys_pos, k2c=k2c, cluster_cnt=self._cluster_cnt
        )
        part_neg = self._data_score_part(
            keys=self.keys_neg, k2c=k2c, cluster_cnt=self._cluster_cnt
        )
        return (part_pos + part_neg) / (2 * self._cluster_cnt)

    @staticmethod
    def _data_score_part(
        keys: t.Sequence[Key], k2c: t.Dict[Key, int], cluster_cnt: int
    ) -> float:
        return (
            pd.Series(map(k2c.get, keys)).value_counts().clip(0, cluster_cnt)
            / cluster_cnt
        ).sum()

    @property
    def keys_pos(self) -> t.List[Key]:
        return sorted(self.pos)

    @property
    def keys_neg(self) -> t.List[Key]:
        return sorted(self.neg)

    @property
    def pos_cnt(self) -> int:
        return len(self.pos)

    @property
    def pos_rate(self) -> float:
        return self.pos_cnt / len(self)

    @property
    def neg_cnt(self) -> int:
        return len(self.neg)

    @staticmethod
    def _key_in_val(
        key: Key, validation_fraction: fractions.Fraction, seed: int
    ) -> bool:
        vf = validation_fraction
        h = mmh3.hash(key=key, seed=seed)
        bucket = h % vf.denominator
        return bucket < vf.numerator

    def split(
        self,
        validation_fraction: fractions.Fraction = cfg.VALIDATION_FRACTION,
        keys_to_remove_from_train: t.Set[Key] = frozenset(),
        keys_to_remove_from_validation: t.Set[Key] = frozenset(),
        seed: int = cfg.SEED,
    ) -> t.Tuple["LabeledDataset", "LabeledDataset"]:
        data = dict(
            train=collections.defaultdict(set), val=collections.defaultdict(set)
        )
        for value, keys in ((True, self.pos), (False, self.neg)):
            for key in keys:
                in_val = self._key_in_val(
                    key=key, validation_fraction=validation_fraction, seed=seed
                )
                if in_val and key not in keys_to_remove_from_validation:
                    data["val"][value].add(key)
                elif not in_val and key not in keys_to_remove_from_train:
                    data["train"][value].add(key)

        ds_train = LabeledDataset(
            label=self.label,
            pos=frozenset(data["train"][True]),
            neg=frozenset(data["train"][False]),
        )
        ds_val = LabeledDataset(
            label=self.label,
            pos=frozenset(data["val"][True]),
            neg=frozenset(data["val"][False]),
        )
        if ds_train.pos_rate == 0:
            raise ValueError(
                f"The training split for label={self.label} with seed={seed} has no positive instances."
            )
        if ds_val.pos_rate == 0:
            raise ValueError(
                f"The validation split for label={self.label} with seed={seed} has no positive instances."
            )
        return ds_train, ds_val

    def boostrap_xyk(self, idx: int) -> t.Tuple[np.ndarray, np.ndarray, t.List[Key]]:
        if idx == 0:
            # return the exact dataset on exactly one of the bootstrap
            # this helps with datasets that have a very small number of positives
            return self.x, self.y, self.keys
        np.random.seed(idx)
        n = len(self)
        idxs = np.random.choice(range(n), n)
        x = self.x[idxs]
        y = self.y[idxs]
        keys = np.array(self.keys)[idxs]
        return x, y, list(keys)

    def remove_keys(self, keys: t.Set[Key]) -> "LabeledDataset":
        return LabeledDataset(
            label=self.label,
            pos=frozenset(self.pos - keys),
            neg=frozenset(self.neg - keys),
        )

    def _create_dataset_from_keys(self, keys: t.Collection[Key]) -> "LabeledDataset":
        keys_pos_set = set(self.keys_pos)
        keys_neg_set = set(self.keys_neg)
        pos = frozenset(k for k in keys if k in keys_pos_set)
        neg = frozenset(k for k in keys if k in keys_neg_set)
        return LabeledDataset(
            label=self.label,
            pos=pos,
            neg=neg,
        )

    def sample(
        self, n: int, seed: t.Optional[int] = None, start_idx: int = 0
    ) -> "LabeledDataset":
        np.random.seed(seed)
        keys_perm = np.random.permutation(self.keys)
        idx_last = start_idx + n if n is not None else None
        keys_to_use = keys_perm[start_idx:idx_last]
        return self._create_dataset_from_keys(keys=keys_to_use)

    def sample_stratified(
        self,
        n: int,
        seed: t.Optional[int] = None,
        min_training_pos_rate: float = 0.0,
    ) -> "LabeledDataset":
        n = min(len(self), n)
        np.random.seed(seed)
        cnt_pos = round(max(self.pos_rate, min_training_pos_rate) * n)
        cnt_neg = n - cnt_pos
        pos = np.random.choice(self.keys_pos, replace=False, size=cnt_pos)
        neg = np.random.choice(self.keys_neg, replace=False, size=cnt_neg)
        return self._create_dataset_from_keys(keys=pos + neg)

    def __add__(self, other: "LabeledDataset") -> "LabeledDataset":
        if self.label != other.label:
            raise ValueError(
                f"Cannot add datasets with two different labels: {self.label} and {other.label}"
            )
        return LabeledDataset(
            label=self.label,
            pos=self.pos | other.pos,
            neg=self.neg | other.neg,
        )

    def __sub__(self, other: "LabeledDataset") -> "LabeledDataset":
        if self.label != other.label:
            raise ValueError(
                f"Cannot subtract datasets with two different labels: {self.label} and {other.label}"
            )
        return LabeledDataset(
            label=self.label,
            pos=self.pos - other.pos,
            neg=self.neg - other.neg,
        )

    def __eq__(self, other: "LabeledDataset") -> bool:
        return (
            len(self) == len(other)
            and self.label == other.label
            and self.pos == other.pos
            and self.neg == other.neg
        )


@dataclass(frozen=True)
class DatasetOrdered:
    label: str
    pos: t.Tuple[Key, ...]
    neg: t.Tuple[Key, ...]

    def __post_init__(self):
        if len(set(self.pos) & set(self.neg)) > 0:
            raise ValueError(
                f"trying to create an ordered dataset with common pos and neg keys."
            )

    def __add__(self, other: "DatasetOrdered"):
        assert other.label == self.label
        pos_set = set(self.pos)
        neg_set = set(self.neg)
        other_pos = tuple(x for x in other.pos if x not in pos_set)
        other_neg = tuple(x for x in other.neg if x not in neg_set)
        return DatasetOrdered(
            label=self.label, pos=self.pos + other_pos, neg=self.neg + other_neg
        )

    @staticmethod
    def _sub(a: tuple, b: tuple) -> tuple:
        b_set = set(b)
        return tuple(x for x in a if x not in b_set)

    def __sub__(self, other: "DatasetOrdered") -> "DatasetOrdered":
        assert other.label == self.label
        pos = self._sub(a=self.pos, b=other.pos)
        neg = self._sub(a=self.neg, b=other.neg)
        return DatasetOrdered(label=self.label, pos=pos, neg=neg)

    @property
    def lds(self) -> LabeledDataset:
        return LabeledDataset(
            label=self.label,
            pos=frozenset(self.pos),
            neg=frozenset(self.neg),
        )

    def __len__(self) -> int:
        return len(self.pos) + len(self.neg)

    @classmethod
    def from_labeled_dataset(
        cls, lds: LabeledDataset, seed: int = cfg.SEED
    ) -> "DatasetOrdered":
        np.random.seed(seed)
        pos = sorted(lds.pos)
        neg = sorted(lds.neg)
        return cls(
            label=lds.label,
            pos=tuple(np.random.permutation(pos)),
            neg=tuple(np.random.permutation(neg)),
        )

    def get_n(self, n: int) -> "DatasetOrdered":
        pos_rate = len(self.pos) / len(self)
        cnt_pos = min(len(self.pos), max(round(pos_rate * n), 1))
        cnt_neg = min(len(self.neg), n - cnt_pos)
        return DatasetOrdered(
            label=self.label,
            pos=self.pos[:cnt_pos],
            neg=self.neg[:cnt_neg],
        )


@dataclass(frozen=True)
class CompDataset:
    label: str
    keys_to_remove: t.FrozenSet[Key] = frozenset()

    @property
    @functools.lru_cache()
    def data(self) -> dict:
        data_og = json.load(open(io.LabelPaths(label=self.label).path_cmp))
        return {
            k: [v for v in vs if v["key"] not in self.keys_to_remove]
            for k, vs in data_og.items()
        }

    def _get_lds(
        self,
        which: str,
        n: t.Optional[int],
        start_idx: int,
        keys_to_remove: t.Optional[t.Set[Key]] = None,
        fail_if_less_available: bool = False,
    ) -> LabeledDataset:
        d = self.data[which]
        if n is not None and n > len(d):
            msg = f"Dataset for label={self.label} has {len(d)} records, but requested {n}"
            if fail_if_less_available:
                raise ValueError(msg)
            else:
                logger.warning(msg)
        data = d[start_idx : start_idx + n] if n is not None else d[start_idx:]
        pos, neg = self.__get_pos_neg(data=data, keys_to_remove=keys_to_remove)
        return LabeledDataset(label=self.label, pos=frozenset(pos), neg=frozenset(neg))

    @staticmethod
    def __get_pos_neg(data, keys_to_remove: t.Optional[t.Set[Key]]):
        keys_to_remove = keys_to_remove if keys_to_remove is not None else set()
        data = [d for d in data if d["key"] not in keys_to_remove]
        pos = (d["key"] for d in data if d["value"])
        neg = (d["key"] for d in data if not d["value"])
        return pos, neg

    def random(
        self,
        n: t.Optional[int] = None,
        start_idx: int = 0,
        keys_to_remove: t.Optional[t.Set[Key]] = None,
    ) -> LabeledDataset:
        return self._get_lds(
            which="random", n=n, start_idx=start_idx, keys_to_remove=keys_to_remove
        )

    def zero_shot(
        self,
        n: t.Optional[int] = None,
        start_idx: int = 0,
        keys_to_remove: t.Optional[t.Set[Key]] = None,
    ) -> LabeledDataset:
        return self._get_lds(
            which="zero_shot", n=n, start_idx=start_idx, keys_to_remove=keys_to_remove
        )

    @property
    def cnt_zero_shot(self) -> int:
        return len(self.data["zero_shot"])

    @property
    def cnt_random(self) -> int:
        return len(self.data["random"])

    def get_ordered_dataset_zero_shot(
        self,
        keys_to_remove: t.Set[Key] = frozenset(),
    ) -> DatasetOrdered:
        return self._get_ordered_dataset(
            which="zero_shot", keys_to_remove=keys_to_remove
        )

    def get_ordered_dataset_random(
        self, keys_to_remove: t.Set[Key] = frozenset()
    ) -> DatasetOrdered:
        return self._get_ordered_dataset(which="random", keys_to_remove=keys_to_remove)

    def _get_ordered_dataset(
        self, which: str, keys_to_remove: t.Set[Key]
    ) -> DatasetOrdered:
        data = self.data[which]
        pos, neg = self.__get_pos_neg(data=data, keys_to_remove=keys_to_remove)
        return DatasetOrdered(label=self.label, pos=tuple(pos), neg=tuple(neg))


def get_aggregate_labeled_dataset(label: str) -> LabeledDataset:
    path = io.LabelPaths(label=label).path_agg
    data = json.load(open(path))
    ds = collections.defaultdict(set)
    for x in data["data"]:
        ds[x["agg"]].add(x["key"])
    return LabeledDataset(
        label=label, pos=frozenset(ds[True]), neg=frozenset(ds[False])
    )


def get_labeled_dataset_positive_cnt_by_key(label: str) -> t.Dict[Key, int]:
    path = io.LabelPaths(label=label).path_agg
    data = json.load(open(path))
    return {x["key"]: x["pos_cnt"] for x in data["data"]}


def get_labeled_dataset_agreement(
    label: str, _annotator_cnt: int = cfg.ANNOTATOR_CNT
) -> float:
    pos_counts = get_labeled_dataset_positive_cnt_by_key(label=label)
    den = len(pos_counts)
    num = sum(cnt in {0, _annotator_cnt} for _, cnt in pos_counts.items())
    return num / den


def get_labeled_dataset_checkpoints(label: str) -> t.List[LabeledDataset]:
    path = io.LabelPaths(label=label).path_checkpoints
    data = json.load(open(path))
    return [
        LabeledDataset(
            label=label,
            pos=frozenset(x["pos"]),
            neg=frozenset(x["neg"]),
        )
        for x in data
    ]


def get_ave_validation_labeled_dataset(label: str) -> LabeledDataset:
    if label not in cfg.LABELS_AVE:
        raise ValueError(
            f"AVE data for label={label} does not exist. choices are: {cfg.LABELS_AVE}"
        )
    path = io.LabelPaths(label=label).path_ave
    data = json.load(open(path))
    return LabeledDataset(
        label=label,
        pos=frozenset(data["pos"]),
        neg=frozenset(data["neg"]),
    )


@dataclass(frozen=False)
class DatasetManager:
    label: str
    _seed: int = cfg.SEED

    def __post_init__(self):
        cmp = CompDataset(label=self.label)
        self._agg = get_aggregate_labeled_dataset(label=self.label)
        self._agg_val_keys = set(self._agg.split(seed=self._seed)[1].keys)
        self._ordered_datasets: t.Dict[str, t.Optional[DatasetOrdered]] = dict(
            agg=self._get_combined_agg_ordered_dataset(),
            zero_shot=cmp.get_ordered_dataset_zero_shot(
                keys_to_remove=self._agg_val_keys
            ),
            random=cmp.get_ordered_dataset_random(keys_to_remove=self._agg_val_keys),
        )
        self.lds = None
        self.lds_hist = []

    def has_data(self, which: str) -> bool:
        return self._ordered_datasets[which] is not None

    def _get_agg_ordered_dataset(self, lds: LabeledDataset) -> DatasetOrdered:
        ods = DatasetOrdered.from_labeled_dataset(lds=lds)
        keys = ods.pos + ods.neg
        pos = tuple(
            k for k in keys if k in self._agg.pos and k not in self._agg_val_keys
        )
        neg = tuple(
            k for k in keys if k in self._agg.neg and k not in self._agg_val_keys
        )
        return DatasetOrdered(label=self.label, pos=pos, neg=neg)

    def _get_combined_agg_ordered_dataset(self) -> DatasetOrdered:
        lds_list = get_labeled_dataset_checkpoints(label=self.label)
        ods = self._get_agg_ordered_dataset(lds=lds_list[0])
        for lds in lds_list[1:]:
            if len(lds) > len(ods):
                ods_curr = self._get_agg_ordered_dataset(lds=lds)
                ods = ods + (ods_curr - ods)
        return ods

    def extend(self, which: str, n: int) -> LabeledDataset:
        ods = self._ordered_datasets[which]
        if ods is not None:
            ods_new = ods.get_n(n=n)
            if len(ods_new) < n:
                logger.warning(
                    f"label={self.label} {which} returned {len(ods_new)} vs. the requested {n}."
                )
                self._ordered_datasets[which] = None
            elif len(ods_new) == len(ods):
                self._ordered_datasets[which] = None
            else:
                self._ordered_datasets[which] = ods - ods_new
            lds_new = ods_new.lds

            self.lds_hist.append(
                dict(which=which, n=n, lds_new_cnt=len(lds_new), lds_new=lds_new)
            )
            self.lds = self.lds + lds_new if self.lds is not None else lds_new
            for name, ods in self._ordered_datasets.items():
                if ods is not None:
                    self._ordered_datasets[name] = (
                        self._ordered_datasets[name] - ods_new
                    )
        else:
            logger.warning(
                f"label={self.label} {which} is already exhausted. returning the last dataset."
            )
        return self.lds
