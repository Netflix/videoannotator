# TODO: add text embeddings for top 10k+ common words and store in repo for convenience

from dataclasses import dataclass
from pathlib import Path

from . import config as cfg


def _check_path(func):
    def __check(*args, **kwargs):
        path = func(*args, **kwargs)
        if not path.exists():
            raise FileNotFoundError(f"file {path} does not exist.")
        return path

    return __check


@dataclass(frozen=True)
class _PathsStatic:
    _path_base: str = cfg.PATH_DATA_BASE

    @property
    @_check_path
    def embeddings(self) -> Path:
        return Path(f"{self._path_base}/embeddings.h5")

    @property
    @_check_path
    def shot_data(self) -> Path:
        return Path(f"{self._path_base}/shot-data.csv")

    @property
    @_check_path
    def text_embeddings(self) -> Path:
        return Path(f"{self._path_base}/text-embeddings.json")


@dataclass(frozen=True)
class _PathAVE(_PathsStatic):
    pass  # TODO


@dataclass(frozen=True)
class LabelPaths:
    label: str
    _path_base: str = cfg.PATH_DATA_BASE

    @property
    @_check_path
    def path_agg(self) -> Path:
        return Path(f"{self._path_base}/agg/{self.label}.json")

    @property
    @_check_path
    def path_checkpoints(self) -> Path:
        return Path(f"{self._path_base}/checkpoints/{self.label}.json")

    @property
    @_check_path
    def path_ave(self) -> Path:
        return Path(f"{self._path_base}/ave/validation/{self.label}.json")

    @property
    @_check_path
    def path_cmp(self) -> Path:
        return Path(f"{self._path_base}/cmp/{self.label}.json")


PATHS_STATIC = _PathsStatic()
PATHS_AVE = _PathAVE()
