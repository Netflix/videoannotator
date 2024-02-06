from copy import deepcopy
from dataclasses import dataclass
import logging
import typing as t

import numpy as np
from tqdm.auto import tqdm

from . import config as cfg
from .data import DatasetManager, get_aggregate_labeled_dataset
from .experiments import ExperimentResults, Metric, run_experiment
from .models import LogisticRegression, Model

Choice = str

# TODO: add conf
CHOICES: t.Tuple[str, ...] = ("agg", "zero_shot", "random")
D = 50
EPSILON = 0.2
ALPHA = 0.5


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionProcessResults:
    res_hist: t.Tuple[t.Tuple[str, ExperimentResults], ...]
    choice_hist: t.Tuple[str, ...]

    @property
    def best_metric(self) -> Metric:
        return max(
            (
                (res.average_precision.mean, res.average_precision)
                for _, res in self.res_hist
            ),
            key=lambda x: x[0],
        )[1]

    @property
    def last_metric(self) -> Metric:
        return self.res_hist[-1][1].average_precision

    @property
    def best_idx(self) -> int:
        return max(
            (res.average_precision.mean, idx)
            for idx, (_, res) in enumerate(self.res_hist)
        )[1]


@dataclass(frozen=False)
class _DecisionProcess:
    label: str
    choices: t.Tuple[str, ...] = CHOICES
    d: int = D
    model: Model = LogisticRegression(scoring="average_precision")
    _seed_base: int = cfg.SEED

    @staticmethod
    def _iter_n(n: int, verbose: bool):
        if verbose:
            return tqdm(range(n))
        else:
            return range(n)

    def run(self, n: int, verbose: bool = False) -> DecisionProcessResults:
        self._warmup(verbose=verbose)
        for _ in self._iter_n(n=n, verbose=verbose):
            choice = self._select_outer()
            res = self._act(choice=choice, verbose=verbose)
            self._update(choice=choice, res=res)
            self._choice_hist.append(choice)
            self._res_hist.append((choice, res))
            self.i += 1
        return DecisionProcessResults(
            res_hist=tuple(self._res_hist),
            choice_hist=tuple(self._choice_hist),
        )

    def __post_init__(self):
        self.i = 0
        self._res_hist = []
        self._choice_hist = []
        self._dsm = DatasetManager(label=self.label, _seed=self._seed_base)
        self._choices_exhausted = set()

    @property
    def choices_available(self) -> t.Tuple[str, ...]:
        return tuple(c for c in self.choices if c not in self._choices_exhausted)

    def _select_outer(self) -> Choice:
        if self.i < len(self.choices):
            # try each option once first
            return self.choices[self.i]
        while True:
            c = self._select()
            if c is None:
                raise ValueError(
                    f"{self.__class__.__name__} select returned option None (not allowed) at i={self.i} with "
                    f"params: {self.__dict__}"
                )
            if self._dsm.has_data(which=c):
                return c
            else:
                self._choices_exhausted.add(c)
                if len(self.choices_available) == 0:
                    raise ValueError(
                        f"{self.__class__.__name__} ran out of options to select at i={self.i} with "
                        f"params: {self.__dict__}"
                    )

    def _select(self) -> Choice:
        pass

    def _warmup(self, verbose: bool) -> None:
        pass

    def _act(self, choice: Choice, verbose: bool) -> ExperimentResults:
        _, lds_val = get_aggregate_labeled_dataset(label=self.label).split(
            seed=self._seed_base
        )
        lds_train = self._dsm.extend(which=choice, n=self.d)
        if verbose:
            logger.info(f"Running experiment with choice={choice}")
        return run_experiment(
            model=self.model,
            lds_train=lds_train,
            lds_validation=lds_val,
        )

    def _update(self, choice: Choice, res: ExperimentResults) -> None:
        pass

    @property
    def _seed(self) -> int:
        return self._seed_base + self.i

    @staticmethod
    def extract_metric(res: ExperimentResults) -> float:
        return res.average_precision.mean


@dataclass
class RoundRobin(_DecisionProcess):
    def _select(self) -> Choice:
        return self.choices_available[self.i % len(self.choices_available)]


@dataclass
class Random(_DecisionProcess):
    def _select(self) -> Choice:
        np.random.seed(self._seed)
        return np.random.choice(self.choices_available)


@dataclass
class GreedyOracle(_DecisionProcess):
    def _select_outer(self) -> Choice:
        while True:
            c = self._select()
            if self._dsm.has_data(which=c):
                return c
            else:
                self._choices_exhausted.add(c)
                if len(self.choices_available) == 0:
                    raise ValueError(
                        f"GreedyOracle has no more available choices for label={self.label} at i={self.i}"
                    )

    def _select(self) -> Choice:
        best, best_score = None, float("-inf")
        dsm_copy = deepcopy(self._dsm)
        for choice in self.choices_available:
            try:
                res = self._act(choice=choice, verbose=False)
                score = self.extract_metric(res=res)
                if np.isnan(score):
                    raise ValueError("Evaluation score is NaN.")
                if score > best_score:
                    best, best_score = choice, score
            except Exception as e:
                logger.warning(
                    f"Running GreedyOracle with choice={choice}, i={self.i}, label={self.label} failed: {e}. "
                    f"Skipping over this choice and setting the metric to zero. "
                    f"Params: {self.__dict__}"
                )
            finally:
                self._dsm = deepcopy(dsm_copy)
        if best is None:
            raise ValueError(
                f"All choices failed for GreedyOracle label={self.label}, i={self.i}."
                f" Params: {self.__dict__}"
            )
        return best


@dataclass
class EpsilonGreedyMean(_DecisionProcess):
    epsilon: float = EPSILON

    def __post_init__(self):
        super().__post_init__()
        self._metrics = {c: Metric(values=tuple()) for c in self.choices}
        self._metrics_abs = {c: Metric(values=tuple()) for c in self.choices}

    @property
    def best_choice(self) -> Choice:
        return max(
            (m.mean, c)
            for c, m in self._metrics.items()
            if c in set(self.choices_available)
        )[1]

    def _select(self) -> Choice:
        np.random.seed(self._seed)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.choices_available)
        else:
            return self.best_choice

    def _update(self, choice: Choice, res: ExperimentResults) -> None:
        prev = self._metrics_abs[self._choice_hist[-1]].values[-1]
        metric = self.extract_metric(res=res)
        self._metrics[choice] = self._metrics[choice].add(metric - prev)
        self._metrics_abs[choice] = self._metrics_abs[choice].add(metric)

    def _warmup(self, verbose: bool) -> None:
        dsm_copy = deepcopy(self._dsm)
        metrics = dict()
        for choice in self.choices:
            try:
                res = self._act(choice=choice, verbose=verbose)
                val = self.extract_metric(res=res)
            except Exception as e:
                val = 0
                name = self.__class__.__name__
                logger.warning(
                    f"Running {name} warmup for choice={choice}, label={self.label} failed: {e}. "
                    f"Skipping over this choice and setting the metric to zero. "
                    f"Params: {self.__dict__}"
                )

            self._metrics_abs[choice] = self._metrics_abs[choice].add(val)
            metrics[choice] = val
            self._dsm = deepcopy(dsm_copy)

        choice_worst_val, choice_worst = min((v, c) for c, v in metrics.items())
        self._choice_hist.append(choice_worst)
        for choice in self.choices:
            self._metrics[choice] = self._metrics[choice].add(
                metrics[choice] - choice_worst_val
            )


@dataclass
class EpsilonGreedyLast(EpsilonGreedyMean):
    @property
    def best_choice(self) -> Choice:
        return max(
            (m.last, c)
            for c, m in self._metrics.items()
            if c in set(self.choices_available)
        )[1]


@dataclass
class EpsilonGreedyEWMA(EpsilonGreedyMean):
    alpha: float = ALPHA

    @property
    def best_choice(self) -> Choice:
        return max(
            (m.ewma(alpha=self.alpha), c)
            for c, m in self._metrics.items()
            if c in set(self.choices_available)
        )[1]


@dataclass
class UCBMean(EpsilonGreedyMean):
    c: float = 1e-2

    """
    c <= 1 for best results?!
    """

    def __post_init__(self):
        super().__post_init__()
        self._counts = {c: 0 for c in self.choices}

    def _select(self) -> Choice:
        t_ = self.i + 1

        best, best_score = None, float("-inf")
        for choice, cnt in self._counts.items():
            if choice in set(self.choices_available):
                if cnt == 0:
                    return choice

                # val + c * sqrt(log(t) / cnt)
                aug = self._metrics[choice].mean
                aug += self.c * np.sqrt(np.log(t_) / cnt)
                if aug > best_score:
                    best = choice
                    best_score = aug

        return best

    def _update(self, choice: Choice, res: ExperimentResults) -> None:
        self._counts[choice] += 1
        super()._update(choice=choice, res=res)


@dataclass
class UCBEWMA(EpsilonGreedyEWMA, UCBMean):
    pass


@dataclass
class UCBLast(EpsilonGreedyLast, UCBMean):
    pass


@dataclass
class Thompson(_DecisionProcess):
    pass


@dataclass
class Exp3Mean(EpsilonGreedyMean):
    gamma: float = 1e-1
    _seed: int = cfg.SEED

    def __post_init__(self):
        assert 0 <= self.gamma <= 1
        super().__post_init__()
        self._ws = {c: 1 for c in self.choices_available}

    @property
    def _k(self) -> int:
        return len(self.choices_available)

    @property
    def _ps(self) -> t.Dict[Choice, float]:
        w_sum = sum(w for c, w in self._ws.items() if c in self.choices_available)
        return {
            c: (1 - self.gamma) * self._ws[c] / w_sum + self.gamma / self._k
            for c in self.choices_available
        }

    def _select(self) -> Choice:
        np.random.seed(self._seed)
        p = [self._ps[c] for c in self.choices_available]
        return np.random.choice(self.choices_available, p=p)

    def _update(self, choice: Choice, res: ExperimentResults) -> None:
        super()._update(choice=choice, res=res)
        x = self._get_update_metric(choice=choice) / self._ps[choice]
        self._ws[choice] *= np.exp(self.gamma * x / self._k)

    def _get_update_metric(self, choice: Choice) -> float:
        return self._metrics[choice].mean


@dataclass
class Exp3Last(Exp3Mean):
    def _get_update_metric(self, choice: Choice) -> float:
        return self._metrics[choice].last


@dataclass
class Exp3EWMA(Exp3Mean):
    alpha: float = ALPHA

    def _get_update_metric(self, choice: Choice) -> float:
        return self._metrics[choice].ewma(alpha=self.alpha)
